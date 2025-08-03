
import numpy as np
from sklearn.linear_model import BayesianRidge
from pathlib import Path
from negmas.preferences import LinearAdditiveUtilityFunction, TableFun
from negmas.outcomes import CategoricalIssue
import csv
from scipy.interpolate import interp1d
from collections import defaultdict

from bayesian_anchor import _dynamic_anchor_bayesian_regression

class OpponentTracker:
    """
    Opponent Bid Tracker - Track, analyze and predict opponent's negotiation behavior
    
    Main functionalities:
    1. Record opponent's bid history and corresponding utility values
    2. Use Bayesian methods to predict opponent's utility function
    3. Predict opponent's next move (utility values and possible bids)
    4. Evaluate prediction accuracy and generate visual analysis
    
    Prediction-related methods:
    - generate_prediction(): Only predict next step utility values, no specific offer generation
    - select_best_offer(): Select best offer based on predicted utility values
    - predict_next_offer(): Combine above methods to predict utility values and recommend offers
    
    Usage example:
    ```python
    # Only predict utility values
    predicted_my_utility, predicted_oppo_utility = tracker.generate_prediction()
    
    # Predict utility values and recommend offer
    best_offer = tracker.predict_next_offer()
    
    # Use custom prediction methods
    best_offer = tracker.predict_next_offer(method_my="bayesian_ridge", method_oppo="monotonic_decreasing")
    ```
    """
    
    def __init__(self, my_ufun, our_name, opp_name, safe_d_path, repeat, negotiator_index):
        """
        Initialize opponent bid tracker
        
        Args:
            my_ufun: Our utility function (used to calculate utility values of opponent bids)
            our_name: Our agent name
            opp_name: Opponent agent name
            safe_d_path: Safe domain path for file naming
            repeat: Repeat number
            negotiator_index: Negotiator index (0=first mover, 1=second mover)
        """
        self.my_ufun = my_ufun
        self.opponent_offers = []  # Record all opponent bids
        self.opponent_utilities = []  # Record utility value of each bid under our utility function
        self.self_offers = []  # Record all our bids
        self.self_utilities = []  # Record our bid utilities under our utility function
        self.bayesian_model = BayesianRidge()  # Bayesian regression model
        self.last_predicted_oppo_utility = None  # Record previous predicted utility value
        self.predicted_opponent_utilities = []  # Record all predicted utility values
        self.predicted_opponent_utilities.append(self.last_predicted_oppo_utility)
        self.predicted_offers = []  # Record predicted offer sets after utility value reverse lookup
        self.history_avg_predicted_offers_utility = []
        self.history_closest_offer_utility = []
        self.history_avg_diff = []
        self.history_closest_diff = []
        self.history_avg_predicted_offers_utility.append(None)
        self.history_closest_offer_utility.append(None)
        self.history_avg_diff.append(None)
        self.history_closest_diff.append(None)
        self.is_diverse_opponent = True
        self.offer_space = list(self.my_ufun.outcome_space.enumerate_or_sample())
        
        # Pre-compute our fixed utility cache in __init__
        self.my_util_cache = {
            offer: float(self.my_ufun(offer))
            for offer in self.offer_space
        }

        # Handle negotiator order
        if negotiator_index == 0:
            # We are first mover, opponent is second mover
            # Use same time sequence index for prediction
            pass 
        else:
            # We are second mover, opponent is first mover (opponent has one more bid)
            # When predicting, opponent should respond to our Index-1 offer
            pass 

        # Store variables for file naming
        self.our_name = our_name
        self.opp_name = opp_name
        self.safe_d_path = safe_d_path
        self.repeat = repeat
        
        # Prediction method settings
        self.method_my = "dynamic_anchor_bayesian_regression"   # Our utility space prediction method
        self.method_oppo = "dynamic_anchor_bayesian_regression"  # Opponent utility space prediction method
        
        # Available methods for method_my (predicting our utility space):
        # dynamic_anchor_bayesian_regression - Default, captures anchoring effects in utility space
        # ensemble - Ensemble multiple models for improved stability and accuracy
        # advanced_gp - Advanced Gaussian Process with confidence intervals
        # adaptive - Auto-select most accurate prediction method with continuous learning
        
        # Available methods for method_oppo (predicting opponent utility space):
        # monotonic_decreasing - Implemented, assumes opponent usually concedes in negotiation
        # pattern_based - Recognizes opponent behavioral patterns
        # concession_model - Optimized for concession-type opponents
        # tft_model - Analyzes relationship between our behavior and opponent reactions
        # adaptive - Dynamically selects best method based on historical performance

        self.likelihood_method = 'trend'
        # Available: 'stepwise', 'regression', 'exception', 'trend'
        
        # Initialize opponent utility function prediction
        # 1. Build opponent weights (initial assumption: equal weights)
        self.num_issues = len(self.my_ufun.issues)
        self.oppo_weights = [round(1.0 / self.num_issues, 5)] * self.num_issues

        # 2. Keep same issues
        self.oppo_issues = self.my_ufun.issues.copy()

        # 3. Reverse opponent issue option values (1 - our utility value) to reflect interest conflicts
        self.oppo_values = []
        for issue_vals in self.my_ufun.values:
            reversed_vals = {option: round(1.0 - utility, 5) for option, utility in issue_vals.mapping.items()}
            self.oppo_values.append(TableFun(reversed_vals))

        # 4. Create predicted opponent utility function
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues
        )

        # Initialize tracking variables
        self.history_predicted_next_offer_utility = []
        self.history_utility_diff = []
        self.predicted_my_space_utilities = []  # Predicted utilities in our space
        self.real_my_space_utilities = []       # Real utilities in our space
        self.predicted_oppo_space_utilities = [] # Predicted utilities in opponent space
        self.counter = 0
        self.relative_time = 0
        self.my_util_cache = {
            offer: float(self.my_ufun(offer))
            for offer in self.offer_space
        }

    def record_oppo_offer(self, offer, relative_time):
        """
        Record opponent's bid and update predictions
        
        Args:
            offer: Opponent's bid
            relative_time: Relative time in negotiation (0.0 to 1.0)
            
        Returns:
            tuple: (predicted_my_utility, predicted_oppo_utility)
        """
        self.relative_time = relative_time
        
        # Add opponent's offer to history
        self.opponent_offers.append(offer)
        
        # Calculate utility under our utility function
        opponent_utility = float(self.my_ufun(offer))
        self.opponent_utilities.append(opponent_utility)

        # Update Bayesian model for opponent utility function
        self.bayesian_update_opponent_ufun_7()
        
        # Generate prediction if sufficient data available
        if len(self.opponent_utilities) >= 3:
            predicted_my_utility, predicted_oppo_utility = self.generate_prediction4()
        else:
            predicted_my_utility, predicted_oppo_utility = 0.5, 0.9
            
        # Record real utility vs prediction comparison if predictions exist
        if len(self.opponent_offers) > 1 and len(self.predicted_my_space_utilities) > 0:
            # Record actual utility value of current bid in our utility space
            self.real_my_space_utilities.append(opponent_utility)

        # Calculate and display opponent bid utility in our function
        try:
            utils = float(self.my_ufun(offer))
        except:
            pass
        
        # Calculate opponent bid utility in predicted opponent function if available
        try:
            if hasattr(self, 'opponent_utility_function') and self.opponent_utility_function is not None:
                opponent_utils = float(self.opponent_utility_function(offer))
        except Exception as e:
            pass
            
        return predicted_my_utility, predicted_oppo_utility

    def bayesian_update_opponent_ufun_7(self, sigma=0.1):
        """
        Update opponent utility function using Bayesian approach (version 7)
        Uses dual reinforcement + dynamic likelihood + early control
        
        Args:
            sigma: Noise parameter for likelihood calculation
        """
        # Check sufficient observations
        if len(self.opponent_offers) < 20:
            return

        # Convert offers to numerical format
        offers_history = np.array([
            [issue.values.index(choice) for issue, choice in zip(self.oppo_issues, offer)]
            for offer in self.opponent_offers
        ])

        # Convert to one-hot encoding
        onehot_history = []
        for offer in offers_history:
            offer_onehot = []
            for i, val in enumerate(offer):
                num_choices = len(self.oppo_issues[i].values)
                onehot = [0] * num_choices
                onehot[val] = 1
                offer_onehot.extend(onehot)
            onehot_history.append(offer_onehot)
        onehot_history = np.array(onehot_history)

        # Build predicted utility vector
        predicted_ufun_vector = []
        for issue_weight, issue_vals in zip(self.oppo_weights, self.oppo_values):
            for option in issue_vals.mapping.values():
                predicted_ufun_vector.append(issue_weight * option)

        newest_bid = onehot_history[-1]
        newest_util = np.dot(predicted_ufun_vector, newest_bid)

        # === Likelihood1: High-value region likelihood ===
        trend_len = int(min(len(onehot_history), max(10, 0.05 * len(onehot_history))))
        recent_offers = onehot_history[-trend_len:]
        recent_utils = [np.dot(predicted_ufun_vector, offer) for offer in recent_offers]
        quantile = 90 if self.relative_time < 0.3 else 85
        q = np.percentile(recent_utils, quantile)
        distance = max(0.0001, abs(newest_util - q))
        likelihood1 = np.exp(- (distance ** 2) / (2 * sigma * sigma))

        # === Likelihood2: If our offer1 rejected and opponent proposes offer2, then u2 should > u1 ===
        likelihood2 = 1.0
        if len(self.self_offers) >= 1 and len(self.opponent_offers) >= 2:
            offer1 = self.self_offers[-1]
            offer2 = self.opponent_offers[-1]
            try:
                u1_oppo = float(self.predicted_oppo_ufun(offer1))
                u2_oppo = float(self.predicted_oppo_ufun(offer2))
                delta = u2_oppo - u1_oppo
                likelihood2 = 1.0 / (1.0 + np.exp(-10 * delta))
            except Exception as e:
                pass

        posterior = likelihood1 * likelihood2
        update_strength = posterior

        # === Reinforcement mechanism 1: Our recent 50 offers vs opponent current offer ===
        if len(self.self_offers) >= 1:
            for my_offer in self.self_offers[-50:]:
                try:
                    u1_self = float(self.my_ufun(my_offer))
                    u2_self = float(self.my_ufun(self.opponent_offers[-1]))
                    if abs(u1_self - u2_self) <= 0.05:
                        diff_index = None
                        for i in range(len(my_offer)):
                            if my_offer[i] != self.opponent_offers[-1][i]:
                                if diff_index is not None:
                                    diff_index = None
                                    break
                                diff_index = i
                        if diff_index is not None:
                            option = self.opponent_offers[-1][diff_index]
                            current_val = self.oppo_values[diff_index].mapping[option]
                            boosted_val = current_val + 0.2 * (1.0 - current_val)
                            self.oppo_values[diff_index].mapping[option] = round(0.8 * current_val + 0.2 * boosted_val, 5)
                except Exception as e:
                    pass

        # === Early Prior: Apply only once in early stage ===
        if not hasattr(self, "early_prior_applied") and self.relative_time < 0.1:
            early_len = max(3, int(0.1 * len(self.opponent_offers)))
            early_offers = offers_history[:early_len]
            for issue_idx, issue in enumerate(self.oppo_issues):
                choices = early_offers[:, issue_idx]
                choice_counts = np.bincount(choices, minlength=len(issue.values))
                choice_freq = choice_counts / choice_counts.sum()
                for val_idx, val in enumerate(issue.values):
                    if choice_freq[val_idx] > 0.2:
                        self.oppo_values[issue_idx].mapping[val] = 0.9
            self.early_prior_applied = True

        # === Update issue weights ===
        stability = np.std(offers_history, axis=0)
        stability_weights = np.exp(-stability)
        self.oppo_weights = stability_weights / stability_weights.sum()

        # === Update option values ===
        for issue_idx, issue in enumerate(self.oppo_issues):
            choices = offers_history[:, issue_idx]
            choice_counts = np.bincount(choices, minlength=len(issue.values))
            choice_freq = choice_counts / choice_counts.sum()

            for val_idx, val in enumerate(issue.values):
                current_val = self.oppo_values[issue_idx].mapping[val]
                delta = choice_freq[val_idx] - current_val
                raw_val = current_val + update_strength * delta
                raw_val = min(max(raw_val, 0.0), 1.0)
                max_growth = 0.15
                adjusted_val = min(raw_val, current_val + max_growth)
                smooth_rate = 0.2
                adjusted_val = (1 - smooth_rate) * current_val + smooth_rate * adjusted_val
                self.oppo_values[issue_idx].mapping[val] = round(adjusted_val, 5)

        # === Rebuild predicted utility function ===
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues
        )

    def get_oppo_offer_history(self):
        """
        Get all opponent bid records
        
        Returns:
            list: List of (offer, utility) tuples for opponent bids
        """
        return list(zip(self.opponent_offers, self.opponent_utilities))

    def get_oppo_last_offer(self):
        """
        Get opponent's last bid and its utility
        
        Returns:
            tuple: (offer, utility) or None if no bids yet
        """
        if self.opponent_offers:
            return self.opponent_offers[-1], self.opponent_utilities[-1]
        return None
    
    def record_self_offer(self, offer):
        """
        Record our bid and calculate its utility under our utility function
        
        Args:
            offer: Our bid
        """
        if offer is not None:
            utility = float(self.my_ufun(offer))
            self.self_offers.append(offer)
            self.self_utilities.append(utility)
            
    def get_self_offer_history(self):
        """
        Get all our bid records
        
        Returns:
            list: List of (offer, utility) tuples for our bids
        """
        return list(zip(self.self_offers, self.self_utilities))

    def get_self_last_offer(self):
        """
        Get our last bid and its utility
        
        Returns:
            tuple: (offer, utility) or None if no bids yet
        """
        if self.self_offers:
            return self.self_offers[-1], self.self_utilities[-1]
        return None

    def generate_prediction4(self, method_my=None, method_oppo=None):
        """
        Generate next step prediction (based on utility value prediction and offer space reverse lookup)
        
        Args:
            method_my: Method for our utility space prediction
            method_oppo: Method for opponent utility space prediction
            
        Returns:
            tuple: (predicted_my_utility, predicted_oppo_utility)
        """
        if len(self.opponent_utilities) < 5:
            return None, None

        if method_my is None:
            method_my = self.method_my
        if method_oppo is None:
            method_oppo = self.method_oppo
        if isinstance(method_my, tuple):
            method_my = method_my[0]
        if isinstance(method_oppo, tuple):
            method_oppo = method_oppo[0]

        # Control maximum window length, default max 40 records
        raw_min_length = min(len(self.opponent_utilities), len(self.self_utilities))
        max_window_size = 40
        if self.relative_time <= 0.3:
            min_length = raw_min_length
        else:
            min_length = raw_min_length // 2
        min_length = max(3, min(min_length, max_window_size))

        # ========== Our perspective prediction ========== #
        utility_diffs_my_space = [
            self.opponent_utilities[i] - self.self_utilities[i]
            for i in range(min_length - 1)
        ]
        X_train_my = np.column_stack((
            range(min_length - 1),
            self.opponent_utilities[:min_length - 1],
            self.self_utilities[:min_length - 1],
            utility_diffs_my_space
        ))
        y_train_my = self.opponent_utilities[1:min_length]
        last_diff_my = self.opponent_utilities[min_length - 1] - self.self_utilities[min_length - 1]
        X_test_my = np.array([[min_length,
                            self.opponent_utilities[min_length - 1],
                            self.self_utilities[min_length - 1],
                            last_diff_my]])

        predicted_my_utilities = _dynamic_anchor_bayesian_regression(X_train_my, y_train_my, X_test_my)
        predicted_my_utilities = predicted_my_utilities[:, 0]

        # ========== Opponent perspective prediction ========== #
        opponent_offers_oppo_space = [float(self.predicted_oppo_ufun(offer)) for offer in self.opponent_offers]
        self_offers_oppo_space = [float(self.predicted_oppo_ufun(offer)) for offer in self.self_offers]
        utility_diffs_oppo_space = [
            opponent_offers_oppo_space[i] - self_offers_oppo_space[i]
            for i in range(min_length - 1)
        ]
        X_train_oppo = np.column_stack((
            range(min_length - 1),
            opponent_offers_oppo_space[:min_length - 1],
            self_offers_oppo_space[:min_length - 1],
            utility_diffs_oppo_space
        ))
        y_train_oppo = opponent_offers_oppo_space[1:min_length]
        last_diff_oppo = opponent_offers_oppo_space[min_length - 1] - self_offers_oppo_space[min_length - 1]
        X_test_oppo = np.array([[min_length,
                                opponent_offers_oppo_space[min_length - 1],
                                self_offers_oppo_space[min_length - 1],
                                last_diff_oppo]])

        predicted_oppo_utilities = _dynamic_anchor_bayesian_regression(
            X_train_oppo, y_train_oppo, X_test_oppo, is_opponent_view=True)
        predicted_oppo_utilities = predicted_oppo_utilities[:, 0]

        # ========== Offer space reverse lookup (intersection filtering) ========== #
        oppo_util_cache = {
            offer: float(self.predicted_oppo_ufun(offer))
            for offer in self.offer_space
        }

        candidate_my = find_candidates_by_bucket(self.my_util_cache, predicted_my_utilities, epsilon=0.05)
        candidate_oppo = find_candidates_by_bucket(oppo_util_cache, predicted_oppo_utilities, epsilon=0.05)
        intersect_offers = list(set(candidate_my) & set(candidate_oppo))

        # ========== Final utility value determination ========== #
        if intersect_offers:
            predicted_my_utility = np.mean([self.my_ufun(o) for o in intersect_offers])
            predicted_oppo_utility = np.mean([self.predicted_oppo_ufun(o) for o in intersect_offers])
        else:
            predicted_my_utility = np.mean(predicted_my_utilities)
            predicted_oppo_utility = np.mean(predicted_oppo_utilities)

        # ========== Store results ========== #
        self.predicted_my_space_utilities.append(predicted_my_utility)
        self.predicted_oppo_space_utilities.append(predicted_oppo_utility)
        self.predicted_offer_candidates = intersect_offers

        return predicted_my_utility, predicted_oppo_utility


def find_candidates_by_bucket(util_cache, predicted_utilities, epsilon=0.05):
    """
    Find candidate offers by bucketing utility values
    
    Args:
        util_cache: Dictionary mapping offers to utility values
        predicted_utilities: List of predicted utility values
        epsilon: Bucket size for grouping similar utilities
        
    Returns:
        list: List of candidate offers
    """
    def bucketize(val):
        return int(val / epsilon)
    
    # Predicted value bucket set
    buckets = set(bucketize(u) for u in predicted_utilities)
    
    # Build utility -> offer reverse index
    bucket_map = defaultdict(list)
    for offer, u in util_cache.items():
        bucket_map[bucketize(u)].append(offer)
    
    # Collect candidate offers
    candidates = []
    for b in buckets:
        candidates.extend(bucket_map.get(b, []))
    
    return candidates
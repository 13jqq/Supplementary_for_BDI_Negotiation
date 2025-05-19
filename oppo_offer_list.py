
import numpy as np
from sklearn.linear_model import BayesianRidge
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import make_interp_spline
from negmas.preferences import LinearAdditiveUtilityFunction, TableFun
from negmas.outcomes import CategoricalIssue
from negmas.preferences import pareto_frontier
import csv
from scipy.interpolate import interp1d
from bayesian_anchor import _dynamic_anchor_bayesian_regression

class OpponentTracker:
    """
    Tracks, models, and predicts opponent behavior in multi-issue negotiation.

    Core Functions:
    - Record and evaluate opponent bids.
    - Predict opponent utility function using Bayesian updates.
    - Predict future offer utilities from both self and opponent perspectives.
    - Provide offer recommendations and evaluate prediction accuracy.
    """
    def __init__(self, my_ufun, our_name, opp_name, safe_d_path, repeat, opponent_ufun, negotiator_index):
        """
        Initialize the opponent tracker.
        :param my_ufun: Utility function of our agent.
        :param opponent_ufun: Real utility function of the opponent (for evaluation purposes).
        """
        self.my_ufun = my_ufun
        self.opponent_offers = []
        self.opponent_utilities = []
        self.self_offers = []
        self.self_utilities = []
        self.bayesian_model = BayesianRidge()
        self.last_predicted_oppo_utility = None
        self.predicted_opponent_utilities = [self.last_predicted_oppo_utility]
        self.opponent_real_utilities = opponent_ufun
        self.current_real_oppo_utility = None
        self.predcited_offers = []
        self.history_real_next_offer_utility = [None]
        self.history_avg_predicted_offers_utility = [None]
        self.history_closest_offer_utility = [None]
        self.history_avg_diff = [None]
        self.history_closest_diff = [None]
        self.is_diverse_opponent = True
        self.offer_space = list(self.my_ufun.outcome_space.enumerate_or_sample())
        self.my_util_cache = {offer: float(self.my_ufun(offer)) for offer in self.offer_space}

        if negotiator_index == 0:
            pass
        else:
            pass

        self.our_name = our_name
        self.opp_name = opp_name
        self.safe_d_path = safe_d_path
        self.repeat = repeat
        self.method_my = "dynamic_anchor_bayesian_regression"
        self.method_oppo = "dynamic_anchor_bayesian_regression"
        self.likelihood_method = 'trend'

        self.num_issues = len(self.my_ufun.issues)
        self.oppo_weights = [round(1.0 / self.num_issues, 5)] * self.num_issues
        self.oppo_issues = self.my_ufun.issues.copy()
        self.oppo_values = []
        for issue_vals in self.my_ufun.values:
            reversed_vals = {option: round(1.0 - utility, 5) for option, utility in issue_vals.mapping.items()}
            self.oppo_values.append(TableFun(reversed_vals))
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues
        )

        self.history_real_next_offer_utility = []
        self.history_predicted_next_offer_utility = []
        self.history_utility_diff = []
        self.predicted_my_space_utilities = []
        self.real_my_space_utilities = []
        self.predicted_oppo_space_utilities = []
        self.real_oppo_space_utilities = []
        self.counter = 0
        self.relative_time = 0
        self.my_util_cache = {
            offer: float(self.my_ufun(offer))
            for offer in self.offer_space
        }


    def record_oppo_offer(self, offer,relative_time):
         """
        Record a new opponent offer and update predictions accordingly.

        Args:
            offer (tuple): The opponent's offer represented as a tuple of selected options.
            relative_time (float): The relative negotiation time (from 0 to 1).

        Returns:
            predicted_my_utility (float): Predicted utility of the next opponent offer from our own utility function.
            predicted_oppo_utility (float): Predicted utility of the next opponent offer from the predicted opponent utility function.
        """
        
        self.relative_time = relative_time
        """
        记录对手的出价
        """
        # Add the opponent's offer to the list
        self.opponent_offers.append(offer)
        
        # Calculate utilities
        opponent_utility = float(self.my_ufun(offer))
        self.opponent_utilities.append(opponent_utility)

        
        self.bayesian_update_opponent_ufun_7()
        
        
        if len(self.opponent_utilities) >= 3:
            predicted_my_utility, predicted_oppo_utility = self.generate_prediction4()
            
            
        else:
            predicted_my_utility, predicted_oppo_utility = 0.5,0.9
            
        
        
        if len(self.opponent_offers) > 1 and len(self.predicted_my_space_utilities) > 0:
            
            self.real_my_space_utilities.append(opponent_utility)
            
           
            real_oppo_utility = float(self.opponent_real_utilities(offer))
            self.real_oppo_space_utilities.append(real_oppo_utility)
       
        
       
      
        utils = float(self.my_ufun(offer))
         
        
       
        try:
            if self.opponent_real_utilities is not None:
                opponent_utils = float(self.opponent_real_utilities(offer))
                
            elif self.opponent_utility_function is not None:
                opponent_utils = float(self.opponent_utility_function(offer))
                
        except Exception as e:
            print(f"{e}")
        return predicted_my_utility, predicted_oppo_utility

    def bayesian_update_opponent_ufun_7(self, sigma=0.1):

        """
        Update the opponent's utility function using Bayesian inference (version 7).

        This version includes multiple likelihood functions (e.g., utility consistency, concession behavior) and reinforcement mechanisms to adjust issue weights and option values.

        Args:
            sigma (float): Standard deviation for the Gaussian likelihood function, controlling sensitivity to utility difference.

        Returns:
            None
        """
        if len(self.opponent_offers) < 20:
            ## print("not enough data")
            return

        offers_history = np.array([
            [issue.values.index(choice) for issue, choice in zip(self.oppo_issues, offer)]
            for offer in self.opponent_offers
        ])

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

        predicted_ufun_vector = []
        for issue_weight, issue_vals in zip(self.oppo_weights, self.oppo_values):
            for option in issue_vals.mapping.values():
                predicted_ufun_vector.append(issue_weight * option)

        newest_bid = onehot_history[-1]
        newest_util = np.dot(predicted_ufun_vector, newest_bid)

       
        trend_len = int(min(len(onehot_history), max(10, 0.05 * len(onehot_history))))
        recent_offers = onehot_history[-trend_len:]
        recent_utils = [np.dot(predicted_ufun_vector, offer) for offer in recent_offers]
        quantile = 90 if self.relative_time < 0.3 else 85
        q = np.percentile(recent_utils, quantile)
        distance = max(0.0001, abs(newest_util - q))
        likelihood1 = np.exp(- (distance ** 2) / (2 * sigma * sigma))


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
                print(f"{e}")

        posterior = likelihood1 * likelihood2
        update_strength = posterior

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
                    print(f" {e}")

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
            

       
        stability = np.std(offers_history, axis=0)
        stability_weights = np.exp(-stability)
        self.oppo_weights = stability_weights / stability_weights.sum()

        
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

        
        self.predicted_oppo_ufun = LinearAdditiveUtilityFunction(
            values=self.oppo_values,
            weights=self.oppo_weights,
            issues=self.oppo_issues
        )

       



    def generate_prediction2(self, method_my=None, method_oppo=None):
       """
        Predict the next-round utility values and identify candidate offers that match both agents' expected utility ranges.

        Args:
            method_my (str or None): The regression method used for self-side utility prediction. Defaults to class-defined method.
            method_oppo (str or None): The regression method used for opponent-side utility prediction. Defaults to class-defined method.

        Returns:
            predicted_my_utility (float): Predicted utility of next opponent offer from our perspective.
            predicted_oppo_utility (float): Predicted utility of next opponent offer from the predicted opponent perspective.
        """
        if len(self.opponent_utilities) < 5:
            ## print("Insufficient data for prediction, need at least 5 data points")
            return None, None

        if method_my is None:
            method_my = self.method_my
        if method_oppo is None:
            method_oppo = self.method_oppo
        if isinstance(method_my, tuple):
            method_my = method_my[0]
        if isinstance(method_oppo, tuple):
            method_oppo = method_oppo[0]

        min_length = min(len(self.opponent_utilities), len(self.self_utilities))

        if self.relative_time <= 0.3:
            window_size = min_length
        else:
            window_size = min_length // 2
        window_size = max(3, window_size)

      
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

        
        
        predicted_oppo_utilities = _dynamic_anchor_bayesian_regression(X_train_oppo, y_train_oppo, X_test_oppo, is_opponent_view=True)
        predicted_oppo_utilities = predicted_oppo_utilities[:, 0]
       
        oppo_util_cache = {
            offer: float(self.predicted_oppo_ufun(offer))
            for offer in self.offer_space
        }

        candidate_my = find_candidates_by_bucket(self.my_util_cache, predicted_my_utilities, epsilon=0.05)
        candidate_oppo = find_candidates_by_bucket(oppo_util_cache, predicted_oppo_utilities, epsilon=0.05)
        intersect_offers = list(set(candidate_my) & set(candidate_oppo))
        
        if intersect_offers:
            predicted_my_utility = np.mean([self.my_ufun(o) for o in intersect_offers])
            predicted_oppo_utility = np.mean([self.predicted_oppo_ufun(o) for o in intersect_offers])
        else:
            
            predicted_my_utility = np.mean(predicted_my_utilities)
            predicted_oppo_utility = np.mean(predicted_oppo_utilities)

       
        self.predicted_my_space_utilities.append(predicted_my_utility)
        self.predicted_oppo_space_utilities.append(predicted_oppo_utility)
        self.predicted_offer_candidates = intersect_offers
       

        return predicted_my_utility, predicted_oppo_utility



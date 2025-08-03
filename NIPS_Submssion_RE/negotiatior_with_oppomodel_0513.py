from negmas.sao import SAONegotiator, SAOResponse, SAOState, ResponseType
from negmas.preferences import PresortingInverseUtilityFunction
from collections import defaultdict
import random

from oppo_offer_list_ufun7done import OpponentTracker


class BDI_Negotiatior(SAONegotiator):
    """
    BDI (Belief-Desire-Intention) Negotiator
    
    A sophisticated negotiation agent that combines:
    1. Opponent modeling through OpponentTracker
    2. Strategic bidding based on predicted opponent behavior
    3. Dynamic acceptance criteria based on negotiation progress
    4. Candidate offer generation and selection mechanisms
    
    Key Features:
    - Tracks opponent's bidding patterns and learns their utility function
    - Generates candidate offers based on opponent predictions
    - Uses weighted random selection for offer diversity
    - Implements time-dependent acceptance thresholds
    - Handles concession detection and strategic responses
    """
    
    def __init__(self, *args, our_name, opp_name, safe_d_path, repeat, **kwargs):
        """
        Initialize BDI Negotiator
        
        Args:
            our_name: Name of our negotiating agent
            opp_name: Name of opponent agent
            safe_d_path: Safe domain path for file operations
            repeat: Repeat number for experiment tracking
        """
        super().__init__(*args, **kwargs)
        
        # Opponent tracking variables
        self.opponent_times = []  # Track timing of opponent offers
        self.opponent_utilities = []  # Track utility values of opponent offers
        self._past_opponent_rv = 0.0  # Past opponent reservation value
        self._rational = []  # Track rational behavior indicators
        
        # Utility tracking
        self.worst_offer_utility = float("inf")  # Worst utility we've offered
        self.best_oppo_util = 0.0  # Best utility opponent has offered us
        
        # Negotiation tools
        self.sorter = None  # Utility-based offer sorter
        self._received = set()  # Set of received offers
        self._sent = set()  # Set of sent offers
        
        # Agent identification
        self.our_name = our_name
        self.opp_name = opp_name
        self.safe_d_path = safe_d_path
        self.repeat = repeat
        
        # Strategic variables
        self.predicted_concession_count = 0  # Count predicted opponent concessions
        self.candidate_map = defaultdict(list)  # Map from utility to candidate offers
        self.opponent_tracker = None  # Opponent behavior tracker

    def on_negotiation_start(self, state: SAOState):
        """
        Initialize negotiation components when negotiation starts
        
        Args:
            state: Current negotiation state
        """
        super().on_negotiation_start(state)

        # Initialize opponent tracker if not already done
        if self.opponent_tracker is None:
            self.opponent_tracker = OpponentTracker(
                self.ufun,  # Our utility function
                self.our_name, 
                self.opp_name,
                self.safe_d_path, 
                self.repeat,
                self.nmi.negotiator_index(self.id)  # Our negotiator index (0=first, 1=second)
            )

        # Initialize utility-based offer sorter
        if self.sorter is None:
            self.sorter = PresortingInverseUtilityFunction(
                self.ufun, 
                rational_only=True,  # Only consider rational offers
                eps=-1,  # No epsilon constraint
                rel_eps=-1  # No relative epsilon constraint
            )
        
        self.sorter.init()
        
        # Add best possible offer to candidate map
        best_offer = self.sorter.outcomes[-1]  # Highest utility offer
        best_util = float(self.ufun(best_offer))
        self.candidate_map[best_util].append(tuple(best_offer))

    def __call__(self, state: SAOState) -> SAOResponse:
        """
        Main negotiation logic - called each round to generate response
        
        Args:
            state: Current negotiation state
            
        Returns:
            SAOResponse: Either accept opponent offer or reject with counter-offer
        """
        if self.opponent_tracker is None:
            raise ValueError("OpponentTracker not initialized")

        # Update opponent tracking if opponent made an offer
        if state.current_offer is not None:
            # Track best utility opponent has offered us
            self.best_oppo_util = max(self.best_oppo_util, self.ufun(state.current_offer))

        # Generate our counter-offer from candidates
        offer = self.select_offer_from_candidates(self.candidate_map)

        # Check if we should accept opponent's current offer
        if state.current_offer and self.should_accept_offer(state.current_offer, state.relative_time):
            return SAOResponse(ResponseType.ACCEPT_OFFER, state.current_offer)

        # Record our offer and update tracking
        self._sent.add(offer)
        self.opponent_tracker.record_self_offer(offer)
        self.worst_offer_utility = min(self.worst_offer_utility, float(self.ufun(offer)))

        return SAOResponse(ResponseType.REJECT_OFFER, offer)

    def should_accept_offer(self, offer, time: float) -> bool:
        """
        Determine whether to accept opponent's offer based on utility and time
        
        Args:
            offer: Opponent's current offer
            time: Relative time in negotiation (0.0 to 1.0)
            
        Returns:
            bool: True if offer should be accepted
        """
        offer_utility = float(self.ufun(offer))

        # End-game acceptance strategy
        if time >= 1.0:
            if self._sent:
                # Accept if better than our worst offer and above reservation value
                return (offer_utility >= min(self.candidate_map.keys()) and 
                       offer_utility >= self.ufun.reserved_value)
            else:
                # Accept if above reservation value
                return offer_utility >= self.ufun.reserved_value

        # Early game - high standards
        if not self.candidate_map:
            return offer_utility >= 0.9

        # Time-dependent acceptance threshold
        sorted_utils = sorted(self.candidate_map.keys())
        accept_ratio = 0.5 + 0.5 * time  # More lenient as time progresses
        threshold_index = int(len(sorted_utils) * (1.0 - accept_ratio))
        threshold_index = min(threshold_index, len(sorted_utils) - 1)
        threshold = sorted_utils[threshold_index]

        return (offer_utility >= threshold and 
               offer_utility >= self.ufun.reserved_value)

    def generate_candidate_utilities(self, state: SAOState):
        """
        Generate candidate offers based on opponent behavior prediction
        
        Args:
            state: Current negotiation state
        """
        predicted_ufun = self.opponent_tracker.predicted_oppo_ufun
        current_threshold = self.worst_offer_utility
        will_concede = False

        # Check if opponent improved their offer to us
        if state.current_offer:
            current_oppo_util = float(self.ufun(state.current_offer))
            if current_oppo_util > self.best_oppo_util + 1e-4:
                will_concede = True

        # Analyze opponent concession patterns
        if len(self.opponent_tracker.opponent_offers) >= 2 and predicted_ufun:
            prev_offer = self.opponent_tracker.opponent_offers[-2]
            curr_offer = self.opponent_tracker.opponent_offers[-1]
            u1, u2 = predicted_ufun(prev_offer), predicted_ufun(curr_offer)

            # Detect if opponent conceded (decreased their utility)
            if u2 < u1 - 1e-4:
                self.predicted_concession_count += 1

            # Concede if opponent has been conceding consistently or in endgame
            if (self.predicted_concession_count >= 5 or 
                (state.relative_time > 0.85 and u2 < u1 - 1e-4)):
                will_concede = True
                self.predicted_concession_count = 0

        # Add concession offer if we decide to concede
        if will_concede and self.sorter:
            concede_offer = self.sorter.next_worse()  # Get next worse offer from sorter
            if concede_offer:
                u = float(self.ufun(concede_offer))
                self.candidate_map[u].append(tuple(concede_offer))

        # Add opponent's past offers as candidates if they meet criteria
        for offer, my_u, _ in zip(
                self.opponent_tracker.opponent_offers,
                self.opponent_tracker.opponent_utilities,
                self.opponent_tracker.predicted_oppo_space_utilities):
            
            # Skip offers we've already sent
            if offer in self._sent:
                continue

            # Add offers better than our worst offer
            if my_u > self.worst_offer_utility:
                self.candidate_map[my_u].append(tuple(offer))
            # Add offers close to current threshold (exploration)
            elif abs(my_u - current_threshold) <= 0.03:
                self.candidate_map[my_u].append(tuple(offer))

    def select_offer_from_candidates(self, candidate_map) -> tuple:
        """
        Select offer from candidate map using weighted random selection
        
        Args:
            candidate_map: Dictionary mapping utilities to lists of offers
            
        Returns:
            tuple: Selected offer
        """
        predicted_ufun = self.opponent_tracker.predicted_oppo_ufun
        
        # Fallback to random selection if no prediction available
        if not candidate_map or not predicted_ufun:
            return random.choice(list(candidate_map.values()))

        # Flatten all candidate offers
        offers = [o for offer_list in candidate_map.values() for o in offer_list]
        
        # Weight offers by predicted opponent utility (higher = more likely to be accepted)
        weights = [max(1e-6, predicted_ufun(o)) for o in offers]
        total = sum(weights)
        probs = [w / total for w in weights]

        # Use weighted random selection
        return random.choices(offers, weights=probs, k=1)[0]

    def on_negotiation_end(self, state: SAOState):
        """
        Clean up when negotiation ends
        
        Args:
            state: Final negotiation state
        """
        # Prevent multiple evaluations
        if getattr(self, "_evaluated", False):
            return
        self._evaluated = True
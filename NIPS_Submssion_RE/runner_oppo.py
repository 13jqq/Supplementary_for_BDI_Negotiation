import re
import json
import time
import math
import numpy as np
from pathlib import Path
import pandas as pd
import os
import sys
import csv

# Disable CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Add working directory paths
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# Add submodule paths to sys.path
sys.path.append('/workspace/neg_workspace/whole_project/anl_agents/anl')
sys.path.append('/workspace/neg_workspace/whole_project/anl_agents/anl_agents')
sys.path.append('/workspace/neg_workspace/whole_project/anl_agents/negmas')
sys.path.append('/workspace/neg_workspace/whole_project/anl_agents')

try:
    import anl_agents
    import negmas
    print("anl_agents successfully imported!")
except ModuleNotFoundError as e:
    print(f"Error importing anl_agents: {e}")
    print("Current sys.path:", sys.path)

# Import negotiation modules
from negmas.sao.negotiators import (
    AspirationNegotiator,
    BoulwareTBNegotiator,
    ConcederTBNegotiator,
    LinearTBNegotiator,
    NaiveTitForTatNegotiator,
)
from negmas.genius.gnegotiators import (
    Atlas3,
    AgentK,
    AgentGG,
    MetaAgent,
    CUHKAgent,
    GeniusNegotiator,
)
from negmas import SAOMechanism, LinearUtilityAggregationFunction, make_issue
import importlib

# Import agents from anl_agents
from anl_agents import get_agents

# Import custom negotiator
from negotiatior_with_oppomodel_0513 import BDI_Negotiatior as BDI_Negotiatior

from negmas.inout import load_genius_domain
from negmas.preferences import UtilityFunction

start_time = time.time()

def bidding_history(our_name, our_agent, opp_name, opp_agent, d_path, domains_used, repeat):
    """
    Run a single negotiation session between two agents
    
    Args:
        our_name: Name of our agent
        our_agent: Our agent class or Java class name
        opp_name: Name of opponent agent
        opp_agent: Opponent agent class or Java class name
        d_path: Domain path
        domains_used: Dictionary of domains
        repeat: Repeat number (currently fixed to 1)
    
    Returns:
        Dictionary containing negotiation results
    """
    repeat = 1
    print(f"Running bidding_history for {our_name} vs {opp_name} in {d_path} (repeat {repeat})")
    the_path_name = d_path.split('/')[-1].split('.')[0]

    # Set up CSV file for results
    csv_filename = f"{the_path_name}_{our_name}_vs_{opp_name}_r{repeat}.csv"
    csv_folder = Path("/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/Results_BDI_0803/sessions")
    
    if not csv_folder.exists():
        csv_folder.mkdir(parents=True, exist_ok=True)
    
    csv_file = csv_folder / csv_filename

    # Check if result file already exists
    if csv_file.exists():
        print(f"Session {csv_filename} already exists. Skipping...")
        return None
    else:
        # Create placeholder file to prevent duplicate tasks
        csv_file.touch()

    # Set up data folder
    data_folder = Path(f'/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/scenarios/')
    d_path_Path = Path(d_path)
    ufun_names = domains_used[d_path]
    
    if not data_folder.exists():
        assert False, f"Data folder {data_folder} does not exist"
    
    data_folder = data_folder / f'r{repeat}' / opp_name
    data_folder.mkdir(parents=True, exist_ok=True)
    
    save_file = data_folder / (d_path_Path.stem + '.json')
    if save_file.exists():
        return

    # Create figures folder
    figures_folder = Path('/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/Results_BDI_0803/sessions/figures/')
    figures_folder.mkdir(parents=True, exist_ok=True)

    # Load utility functions from profile files
    # Load utility function 1 (Profile A)
    ufun1_file = Path('/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/scenarios/scenarios/special') / d_path_Path.stem / 'profileA.json'
    dic_ufun1_file = json.load(ufun1_file.open(mode='r'))['LinearAdditiveUtilitySpace']
    issues = [negmas.make_issue(name=issue_name, values=values['values']) 
              for issue_name, values in list(dic_ufun1_file['domain']['issuesValues'].items())]
    values_us = {issue_name: values['DiscreteValueSetUtilities']['valueUtilities'] 
                 for issue_name, values in dic_ufun1_file['issueUtilities'].items()}
    ufun1 = negmas.LinearUtilityAggregationFunction(values=values_us, 
                                                   weights=dic_ufun1_file['issueWeights'], 
                                                   issues=issues)
    
    # Load utility function 2 (Profile B)
    ufun2_file = Path('/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/scenarios/scenarios/special') / d_path_Path.stem / 'profileB.json'
    dic_ufun2_file = json.load(ufun2_file.open(mode='r'))['LinearAdditiveUtilitySpace']
    issues = [negmas.make_issue(name=issue_name, values=values['values']) 
              for issue_name, values in list(dic_ufun2_file['domain']['issuesValues'].items())]
    values_us = {issue_name: values['DiscreteValueSetUtilities']['valueUtilities'] 
                 for issue_name, values in dic_ufun2_file['issueUtilities'].items()}
    ufun2 = negmas.LinearUtilityAggregationFunction(values=values_us, 
                                                   weights=dic_ufun2_file['issueWeights'], 
                                                   issues=issues)
    
    safe_d_path = re.sub(r'[^\w\-_\. ]', '_', str(d_path))  # Replace invalid filename characters
    
    print("----------------------")
    print(ufun1)
    print(ufun2)
    print("----------------------")
    
    # Create negotiation session
    session = negmas.SAOMechanism(issues=issues, time_limit=180)
    
    # Create our agent instance
    if isinstance(our_agent, type):  # Python class
        if our_name == 'BDI_Negotiatior':
            our_agent_in = our_agent(
                name=our_name, our_name=our_name, opp_name=opp_name,
                safe_d_path=safe_d_path, repeat=repeat
            )
        else:
            our_agent_in = our_agent(name=our_name)
    else:  # Genius Negotiator with Java class name
        our_agent_in = negmas.genius.GeniusNegotiator(name=our_name, java_class_name=our_agent)

    # Create opponent agent instance
    if isinstance(opp_agent, type):
        if opp_name == 'BDI_Negotiatior':
            opp_agent_in = opp_agent(
                name=opp_name, our_name=our_name, opp_name=opp_name,
                safe_d_path=safe_d_path, repeat=repeat
            )
        else:
            opp_agent_in = opp_agent(name=opp_name)
    else:
        opp_agent_in = negmas.genius.GeniusNegotiator(name=opp_name, java_class_name=opp_agent)

    print("our_agent_in: ", our_agent_in)
    print("opp_agent_in: ", opp_agent_in)
    
    # Add agents to session with their preferences
    session.add(our_agent_in, preferences=ufun1)
    session.add(opp_agent_in, preferences=ufun2)
    
    # Run the negotiation
    session.run()
    
    # Get negotiation results
    agreement = session.agreement
    print(f"agreement: {agreement}")
    
    # Calculate utilities for both agents
    self_ufun_agreement = ufun1(agreement) if agreement else 0
    print(f"self_ufun_agreement: {self_ufun_agreement}")
    oppo_ufun_agreement = ufun2(agreement) if agreement else 0
    print(f"oppo_ufun_agreement: {oppo_ufun_agreement}")

    # Calculate Pareto and Nash distances
    pareto_frontier = session.pareto_frontier(max_cardinality=float('inf'), sort_by_welfare=True)
    frontier = pareto_frontier[0]
    pareto_distance = calculate_pareto_distance([self_ufun_agreement, oppo_ufun_agreement], 
                                              frontier, [ufun1, ufun2])
    
    nash_point = session.nash_points(max_cardinality=float('inf'))[0]
    nash_distance = calculate_nash_distance([self_ufun_agreement, oppo_ufun_agreement], 
                                          nash_point[0], [ufun1, ufun2])

    # Prepare results record
    number_of_steps = len(session.trace)
    total_social_welfare = self_ufun_agreement + oppo_ufun_agreement
    
    record = {
        'self_agent': our_name,
        'opp_agent': opp_name,
        'env': the_path_name,
        'agreement': agreement,
        'self_ufun_agreement': self_ufun_agreement,
        'oppo_ufun_agreement': oppo_ufun_agreement,
        'number_of_steps': number_of_steps,
        'total_social_welfare': total_social_welfare,
        'pareto_distance': pareto_distance,
        'nash_distance': nash_distance
    }

    # Write results to CSV file
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=record.keys())
        writer.writeheader()
        writer.writerow(record)
    
    return record


def calculate_pareto_distance(agreement_utility, pareto_frontier, ufuns):
    """Calculate minimum distance from agreement utility to Pareto frontier"""
    pareto_distance = float("inf")
    cu = agreement_utility
    for pu in pareto_frontier:
        dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
        if dist < pareto_distance:
            pareto_distance = dist
    return pareto_distance


def calculate_nash_distance(agreement_utility, nash_point, ufuns):
    """Calculate distance from agreement utility to Nash point"""
    cu = agreement_utility
    nash_distance = math.sqrt((nash_point[0] - cu[0]) ** 2 + (nash_point[1] - cu[1]) ** 2)
    return nash_distance


if __name__ == '__main__':
    # Load domains to use for training
    domain4train_file = Path('/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/scenarios/test_use.json')
    sessions_folder = Path("/workspace/Oppo_Model/BDI/scenarios/Results_BDI_0803/sessions")
    domains_used = json.load(domain4train_file.open(mode='r'))
    
    # Define agent names
    agent_name = [
    'BDI_Negotiatior', 'TheFawkes', 'Atlas3', 'CUHKAgent', 'ParsAgent',
    'RandomDance', 'TMFAgent', 'MiCRO', 'PonPokoAgent', 'MetaAgent2013',
     'BoulwareTBNegotiator', 'ConcederTBNegotiator','NaiveTitForTatNegotiator']
    my_agents = {'BDI_Negotiatior'}
    opponent_agents = set(agent_name) - my_agents

    # Load opponent agent configurations
    json_file_path = Path(__file__).parent / 'opponents_genius_BDI.json'
    oppo_names = json.load(json_file_path.open(mode='r'))
    
    # Create results directory
    results_dir = Path('/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/Results_BDI_0803')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize result arrays
    n = len(agent_name)
    total_ufun_agreement = np.zeros(n)
    total_ufun_oppo_agreement = np.zeros(n)
    number_of_steps = np.zeros(n)
    total_social_welfare = np.zeros(n)
    pareto_distance = np.zeros(n)
    nash_distance = np.zeros(n)
    agreement_num = np.zeros(n)
    agreement_total = np.zeros(n)
    
    num_repeats = 1
    all_results = []

    # Build agent name to class/path mapping
    genius_agent_map = {name: cls_path for name, cls_path in oppo_names}
    
    # Add our custom agents
    genius_agent_map['BDI_Negotiatior'] = BDI_Negotiatior
    genius_agent_map['BoulwareTBNegotiator'] = BoulwareTBNegotiator
    genius_agent_map['ConcederTBNegotiator'] = ConcederTBNegotiator
    genius_agent_map['LinearTBNegotiator'] = LinearTBNegotiator
    genius_agent_map['NaiveTitForTatNegotiator'] = NaiveTitForTatNegotiator

    # Run experiments (single-threaded)
    for repeat in range(num_repeats):
        for i in range(len(agent_name)):
            for j in range(len(agent_name)):
                agent_i_name = agent_name[i]
                agent_j_name = agent_name[j]

                # Only run cases where our agent is BDI_Negotiatior vs opponents
                if agent_i_name not in my_agents or agent_j_name not in opponent_agents:
                    continue

                for d_path in domains_used.keys():
                    our_name = agent_i_name
                    our_agent = genius_agent_map[our_name]
                    opp_name = agent_j_name
                    opp_agent = genius_agent_map[opp_name]

                    print(f'Running: {our_name} vs {opp_name} on {d_path}')
                    
                    # Run single negotiation session
                    result = bidding_history(
                        our_name, our_agent, opp_name, opp_agent, 
                        d_path, domains_used, repeat
                    )

    # Process and aggregate results
    for csv_file in sessions_folder.glob("*.csv"):
        session_data = pd.read_csv(csv_file)
        all_results.append(session_data)
    
    if all_results:
        full_results = pd.concat(all_results, ignore_index=True)

        # Process each record in the DataFrame
        for index, record in full_results.iterrows():
            self_agent = record['self_agent']
            opp_agent = record['opp_agent']

            i = agent_name.index(self_agent)
            j = agent_name.index(opp_agent)

            # Aggregate statistics
            total_ufun_agreement[i] += record['self_ufun_agreement']
            total_ufun_oppo_agreement[i] += record['oppo_ufun_agreement']
            total_ufun_agreement[j] += record['oppo_ufun_agreement']
            total_ufun_oppo_agreement[j] += record['self_ufun_agreement']
            number_of_steps[i] += record['number_of_steps']
            number_of_steps[j] += record['number_of_steps']
            total_social_welfare[i] += record['total_social_welfare']
            total_social_welfare[j] += record['total_social_welfare']
            pareto_distance[i] += record['pareto_distance']
            pareto_distance[j] += record['pareto_distance']
            nash_distance[i] += record['nash_distance']
            nash_distance[j] += record['nash_distance']
            agreement_total[i] += 1
            agreement_total[j] += 1

            if pd.notnull(record['agreement']):
                agreement_num[i] += 1
                agreement_num[j] += 1

        # Calculate average results
        repeat_results = pd.DataFrame({
            'Agent Name': agent_name,
            'UFun': np.divide(total_ufun_agreement, agreement_total, 
                            out=np.zeros_like(total_ufun_agreement), where=agreement_total != 0),
            'UFun Oppo': np.divide(total_ufun_oppo_agreement, agreement_total, 
                                 out=np.zeros_like(total_ufun_oppo_agreement), where=agreement_total != 0),
            'Steps': np.divide(number_of_steps, agreement_total, 
                             out=np.zeros_like(number_of_steps), where=agreement_total != 0),
            'Welfare': np.divide(total_social_welfare, agreement_total, 
                               out=np.zeros_like(total_social_welfare), where=agreement_total != 0),
            'Pareto': np.divide(pareto_distance, agreement_total, 
                              out=np.zeros_like(pareto_distance), where=agreement_total != 0),
            'Nash': np.divide(nash_distance, agreement_total, 
                            out=np.zeros_like(nash_distance), where=agreement_total != 0),
            'ratio': np.divide(agreement_num, agreement_total, 
                             out=np.zeros_like(agreement_num), where=agreement_total != 0)
        })

        # Save aggregated results to final CSV
        full_results_file = Path("/workspace/Oppo_Model/BDI/NIPS_Submssion_RE/Results_BDI_0803/final_results.csv")
        repeat_results.to_csv(full_results_file, index=False)
        print(f"All results saved to {full_results_file}")

    # end_time = time.time()
    # total_time_minutes = (end_time - start_time) / 60
    # print(f"Total time taken for the experiment: {total_time_minutes:.2f} minutes.")

import copy
import json
from tqdm import tqdm
import requests
import numpy as np
import pandas as pd


def url_map(x):
    """
    Standardizes a URL by ensuring it ends with a '/' and does not contain hash fragments.
    """
    if '#' in x:
        x = x[:x.index('#')]
    x = x if x[-1] == '/' else (x + '/')

    return x

def load_judge_data(Config):
    """
    Load judgement data and define relevance score based on position.
    Assumes that article position is in *descending* order of relevance.
    """
    this_dtype = {Config.judge_posn_key: int, Config.judge_query_key: str}
    judge_data = pd.read_csv(Config.judge_filepath, dtype=this_dtype)
    judge_data[Config.judge_url_key] = judge_data[Config.judge_url_key].apply(url_map)
    if Config.compute_relevance_method == "reverse_position":
        # compute relevance by reversing position:
        ## relevance := max(posn) - posn + 1, by query
        judge_data[Config.judge_rlv_key] = judge_data.groupby(Config.judge_query_key)[Config.judge_posn_key].\
            transform('max') - judge_data[Config.judge_posn_key] + 1
    else:
        # adhoc segmentation method, values can be tailored
        judge_data[Config.judge_rlv_key] = 1
        judge_data.loc[judge_data[Config.judge_posn_key] < 10, Config.judge_rlv_key] = 2
        judge_data.loc[judge_data[Config.judge_posn_key] < 6, Config.judge_rlv_key] = 3
        judge_data.loc[judge_data[Config.judge_posn_key] < 3, Config.judge_rlv_key] = 4

    return judge_data



class SearchApiParams(object):
    def __init__(self, Config, query=""):

        self.headers = {
            "api-key": Config.query_api_code,
            "Content-Type": "application/json",
            "Accept": "application/json"}

        self.body =  {
            "scoringProfile": Config.profile_name,
            "queryType": Config.query_type,
            "searchFields": Config.search_fields_str ,
            "select": Config.api_url_key,
            "searchMode": "any",
            "top": str(Config.top_k_srch_results)
        }

    def update_search_term(self, query):
        searchTerms = query.replace("'", "''")
        st_str = ','.join(f"'{a}'" for a in searchTerms.split())
        self.body["search"] = st_str

        return None

def get_api_search_results(Config):
    search_api_params = SearchApiParams(Config=Config)
    df_results = pd.DataFrame([])  # initialize results gathering
    for this_query in Config.selected_judge_queries:
        try:
            search_api_params.update_search_term(query=this_query)
            response = requests.post(url=Config.query_url,
                                     json=search_api_params.body,
                                     headers=search_api_params.headers)
            result = response.json()
            df = pd.DataFrame(result[Config.api_value_key])
            df.columns = [Config.api_score_key, Config.api_url_key]
            df[Config.api_score_key] = df[Config.api_score_key].fillna(0)
            df[Config.api_query_key] = this_query
            df_results = pd.concat([df_results, df])
        except Exception as e:
            if this_query not in Config.exception_queries:
                # avoid logging duplicate exceptions
                print('Exception for querying: %s' % this_query)
                print('Check that %s returns results in your index' % this_query)
                Config.exception_queries.append(this_query)

    return df_results


def dcg_at_k(rlv_vals, k, method=0):
    """Compute ndcg@k of sorted relevance values.

    Params:
        rlv_vals (list or np-like array (float)): ordered relevance values
        k (int): number of top results
        method (int): dcg computation method key

    Returns:
        dcg (float): discounted cumulative gain @ k

    Notes:
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain
    """
    dcg = 0.  # initialize
    rlv_vals = np.asfarray(rlv_vals)[ :k]  # get top k relevance values
    if rlv_vals.size:
        if method == 0:
            dcg = rlv_vals[0] + np.sum(rlv_vals[1:] / np.log2(np.arange(2, rlv_vals.size + 1)))
        elif method == 1:
            dcg = np.sum(rlv_vals / np.log2(np.arange(2, rlv_vals.size + 2)))
        else:
            raise ValueError('dcg method must be 0 or 1.')
    return dcg

def ndcg_at_k(rlv_vals, k, method=0):
    """Compute ndcg@k of sorted relevance values.

    Params:
        rlv_vals (list or np-like array (float)): ordered relevance values
        k (int): number of top results
        method (int): dcg computation method key

    Returns:
        ndcg_data (DataFrame): normalized discounted cumulative gain @ k, per query

    Notes:
        https://en.wikipedia.org/wiki/Discounted_cumulative_gain

    """
    ideal_sorted_list = sorted(rlv_vals, reverse=True)  # most relevant result at first index
    dcg_max = dcg_at_k(rlv_vals=ideal_sorted_list, k=k, method=method)
    if not dcg_max:
        return 0.
    ndcg = dcg_at_k(rlv_vals=rlv_vals, k=k, method=method) / dcg_max

    return ndcg

def compute_queries_ndcg(judge_data, Config):
    """
    Compute nDCG@k for each query sampled from the set of google queries.

    Params:
        fn (implicit function): search results call

    Returns:
        ndcg_data (DataFrame): judgement data merge with returned search results

    Notes:
        Results computed by fetching search results from Azure search instance, per query,
        and computing intersection between results returned and judgement data.

    """
    query_list = np.unique(judge_data[Config.judge_query_key])
    search_results = get_api_search_results(Config=Config)
    # print('KEYS OF SEARCH RESULTS\n', list(search_results))
    search_results[Config.api_url_key] = search_results[Config.api_url_key].apply(url_map)
    # search_results = search_results[~search_results[Config.api_url_key].str.contains(Config.api_no_result_val)]
    merged_result = pd.merge(left=judge_data,
                             right=search_results,
                             how='outer',
                             left_on=[Config.judge_query_key, Config.judge_url_key],  # judgment data
                             right_on=[Config.api_query_key, Config.api_url_key])  # search api results
    merged_result[Config.judge_rlv_key] = merged_result[Config.judge_rlv_key].fillna(0)
    merged_result[Config.api_score_key] = merged_result[Config.api_score_key].fillna(0)
    # compute ndcg per query: ascending=False => most relevant has highest score
    ndcg_data = merged_result.sort_values(by=Config.api_score_key, ascending=False).\
                groupby(by=Config.judge_query_key).\
                relevance.agg(lambda x: ndcg_at_k(x, Config.top_k_srch_results)).\
                reset_index()
    ndcg_data.columns = [Config.judge_query_key, Config.ndcg_key]

    return ndcg_data


def update_scoring_profile(update_dict, Config, debug=False):
    """
    Update an existing scoring profile's parameters via overwrite.

    Params:
        update_dict (dict): key, values to update scoring profile params
        debug (bool): print info to console/stdout

    Returns:
        None
    """
    headers = {
        'api-key': Config.admin_api_code,
        'User-Agent': Config.user_agent,
        'Content-Type': 'application/json',

        }
    try:
        response = requests.get(Config.score_profile_index_url, headers=headers)
        index = response.json()
        updateIndex = -1
        defaultProfileIndex = -1
        for i, profile in enumerate(index['scoringProfiles']):
            if profile['name'] == index['defaultScoringProfile']:
                defaultProfileIndex = i
            if profile['name'] == Config.profile_name:
                if debug:
                    print(f'scoringProfile found at index {i}')
                updateIndex = i
                break
        else:
            if debug:
                print('scoringProfile not found, creating a new one.')
            new_profile = copy.deepcopy(index['scoringProfiles'][defaultProfileIndex])
            new_profile['name'] = Config.profile_name
            index['scoringProfiles'].append(new_profile)

        sp = index['scoringProfiles'][updateIndex]
        del index['@odata.context']
        del index['@odata.etag']

        if debug:
            print('Old scoring profile:', sp)

        update_dict_keys = update_dict.keys()
        if 'weight' in update_dict_keys:
            sp['text']['weights'] = update_dict['weight']
        if 'boost' in update_dict_keys:
            for i in sp['functions']:
                if i['fieldName'] in update_dict['boost']:
                    i['boost'] = update_dict['boost'][i['fieldName']]

        if debug:
            print('New scoring profile: ', sp)
        # Upload new scoring profile to Azure search index
        update_response = requests.put(Config.score_profile_index_url,
                                       headers=headers,
                                       data=json.dumps(index))
        if update_response.status_code != 204:
            print('Error:', update_response.status_code, update_response.text)
            print(index)
    except Exception as e:
        print(response, e)

    return None


def optimization_objective(judge_data, Config):
    """
    Set the optimization objective (calback) function.

    Returns:
        optuna_objective_fn (func)
    """
    def _objective_fn(trial):
        """
        Callback function for optuna optimization

        Params:
            trial (optuna.trial.Trial): implicitly passed by objective fn

        Returns:
            ndcg_avg (float)
        """
        update_dict = dict(
            weight={
                f: trial.suggest_discrete_uniform(f'{f}_weight', Config.min_weight,
                                                  Config.max_weight, Config.weight_steps)
                for f in Config.weight_fields})

        print('setting weights', json.dumps(update_dict))  # updates on stdout, also written to history
        update_scoring_profile(update_dict=update_dict,
                               Config=Config,
                               debug=False)
        result = compute_queries_ndcg(judge_data=judge_data,
                                      Config=Config)


        ndcg_avg = float(result[Config.ndcg_key].mean())
        trial.set_user_attr('params', update_dict)
        trial.set_user_attr('ndcg_avg', ndcg_avg)
        return ndcg_avg

    return _objective_fn
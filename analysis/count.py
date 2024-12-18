'''counting number of pro-life and pro-choice speeches per congressional term'''

import pandas as pd

def congress_to_year(congress_number):
    return 1789 + 2 * (congress_number - 1)

def main():
    data = pd.read_csv('../data/all_predictions.csv')
    count_data = data.groupby('congress')['labeled_class'].value_counts().unstack(fill_value=0)
    
    count_data.columns = ['count_pro_life', 'count_pro_choice']
    
    count_data = count_data.reset_index()
    count_data = count_data.set_index('congress').reindex(range(43, 119), fill_value=0).reset_index()
    
    count_data['year'] = count_data['congress'].apply(congress_to_year)
    
    count_data.to_csv('../data/counts.csv', index=False)
    
    zero_count_congresses = count_data[(count_data['count_pro_life'] == 0) & (count_data['count_pro_choice'] == 0)]
    
    print("\ncongresses with 0 for both counts:")
    print(zero_count_congresses)

if __name__ == '__main__':
    main()

'''creating sentiment, polarization, and public opinion graphs'''

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

def clean_public_data(public_file):
    public = pd.read_csv(public_file, header=2, skipfooter=2, engine='python')
    public.columns = ['Year', 'Legal in all/most cases', 'Illegal in all/most cases']
    public['Legal in all/most cases'] = public['Legal in all/most cases'].replace({'%': '', '--': 'NaN'}, regex=True).astype(float)
    public['Illegal in all/most cases'] = public['Illegal in all/most cases'].replace({'%': '', '--': 'NaN'}, regex=True).astype(float)

    def clean_year(x):
        if isinstance(x, str):
            split_year = x.split()
            if len(split_year) > 1:
                return split_year[-1]
        return x
    
    public['Year'] = public['Year'].apply(clean_year)
    public['Year'] = pd.to_numeric(public['Year'], errors='coerce')
    public.dropna(subset=['Year', 'Legal in all/most cases', 'Illegal in all/most cases'], inplace=True)

    return public
    
def main():
    counts = pd.read_csv('../data/counts.csv')
    public = clean_public_data('../data/prc.csv')
    outdir = '../plots'

    counts['total_speeches'] = counts['count_pro_life'] + counts['count_pro_choice']
    counts['overall_sentiment'] = (counts['count_pro_choice'] - counts['count_pro_life']) / counts['total_speeches']
    counts['polarization'] = abs(counts['overall_sentiment'])
    counts.fillna(0, inplace=True)
    
    counts['pro_choice_percentage'] = (counts['count_pro_choice'] / counts['total_speeches']) * 100
    counts['pro_life_percentage'] = (counts['count_pro_life'] / counts['total_speeches']) * 100

    # overall sentiment
    plt.figure(figsize=(10, 5))
    plt.plot(counts['year'], counts['overall_sentiment'], marker='o', color='green', label='Overall Sentiment')
    plt.title('Overall Sentiment Toward Abortion, 1873-Present', fontsize=25)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Sentiment', fontsize=18)
    plt.axhline(0, color='black', linewidth=2, zorder=2)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    roe_height = .7
    dobbs_height = .85
    roe_y_pos = 0.6
    dobbs_y_pos = 0.63
    label_height = 5
    
    def place_year_labels():
        plt.axvline(1973, color='black', linestyle='--', linewidth=2, zorder=-1)
        plt.axvline(2022, color='black', linestyle='--', linewidth=2, zorder=-1)
    
        plt.gca().add_patch(Rectangle((1973+label_height/2, roe_y_pos - roe_height / 2), roe_height, label_height, color='white', zorder=10, angle=90))
        plt.gca().add_patch(Rectangle((2022+label_height/2, dobbs_y_pos - dobbs_height / 2), dobbs_height, label_height, color='white', zorder=10, angle=90))

        plt.text(1973, roe_y_pos, 'Roe v. Wade', ha='center', va='center', fontsize=12, color='black', zorder=11, rotation=90)
        plt.text(2022, dobbs_y_pos, 'Dobbs v. Jackson', ha='center', va='center', fontsize=12, color='black', zorder=11, rotation=90)
        
    place_year_labels()

    neutral_x = 0.45 * (counts['year'].max() - counts['year'].min()) + counts['year'].min()
    rect_height = 0.14
    rect_width = 14
    plt.gca().add_patch(Rectangle((neutral_x - rect_width / 2, -rect_height / 2), rect_width, rect_height, color='white', zorder=10))
    plt.text(neutral_x, 0, 'Neutral', ha='center', va='center', fontsize=12, color='black', rotation=0, zorder=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'overall_sentiment.png'))
    plt.close()

    # polarization
    plt.figure(figsize=(10, 5))
    plt.plot(counts['year'], counts['polarization'], marker='o', color='purple', label='Polarization')
    plt.title('Polarization on the Topic of Abortion, 1873-Present', fontsize=25)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Polarization', fontsize=18)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)

    roe_y_pos = 0.2
    dobbs_y_pos = 0.6
    roe_height = .36
    dobbs_height = .43
    place_year_labels()
        
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'polarization.png'))
    plt.close()
    
    # compared with public opinion
    counts = counts[counts['year'] >= 1995]
    plt.figure(figsize=(10, 5))
    plt.plot(counts['year'], counts['pro_life_percentage'], marker='o', color='red', label='% Pro-Life of Congress', linestyle='-', linewidth=2)
    plt.plot(public['Year'], public['Legal in all/most cases'], marker='o', color='#6FAFFF', label='% Pro-Choice of the Public', linestyle='--', linewidth=2)
    plt.plot(public['Year'], public['Illegal in all/most cases'], marker='o', color='#FF6F6F', label='% Pro-Life of the Public', linestyle='--', linewidth=2)
    plt.plot(counts['year'], counts['pro_choice_percentage'], marker='o', color='blue', label='% Pro-Choice of Congress', linestyle='-', linewidth=2)

    plt.title('The Public\'s Opinion on Abortion Versus Congress\'s, 1995-Present', fontsize=20)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Percentage', fontsize=18)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    dobbs_y_pos = 76.5
    dobbs_height = 23
    label_height = 1
    plt.axvline(2022, color='black', linestyle='--', linewidth=2, zorder=-1)
    plt.gca().add_patch(Rectangle((2022-label_height/2, dobbs_y_pos-dobbs_height/2), label_height, dobbs_height, color='white', zorder=10))
    plt.text(2022, dobbs_y_pos, 'Dobbs v. Jackson', ha='center', va='center', fontsize=9, color='black', zorder=11, rotation=90)

    plt.legend(loc='upper left', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'public_opinion_vs_legal_status.png'))
    plt.close()

if __name__ == '__main__':
    main()

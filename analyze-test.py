import pandas as pd
import seaborn as sns
import string
from word_forms.lemmatizer import lemmatize
import json
from matplotlib import pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.stats.multicomp import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import numpy as np


def calculate_latin_ratio(word_list):
    latin_count = 0
    word_count = 0
    for word in word_list:
        if word.isdigit() or not any(c.isalpha() for c in word):
            continue
        word.strip("“”\"'-")
        try:
            lemmatized = lemmatize(word).lower()
        except:
            lemmatized = word.lower()
        try:
            is_latin = False
            if lemmatized in word_dict:
                is_latin = word_dict[lemmatized]
            if is_latin:
                latin_count += 1
            word_count += 1
            word_dict[lemmatized] = is_latin
        except:
            pass
    return latin_count / word_count


def build_lr(file):
    translator = str.maketrans("", "", string.punctuation)

    word_dict = {}
    with open("./latin_dict.json", "r") as f:
        word_dict = json.load(f)

    df = pd.DataFrame(pd.read_csv(f"{file}.csv"))

    for i in range(len(df)):
        abstract = df.loc[i, "Abstract"]
        if type(abstract) != str:
            continue
        words = abstract.translate(translator).split()
        latin_ratio = calculate_latin_ratio(words)
        df.loc[i, "Latin Ratio"] = latin_ratio

    df.to_csv(f"{file}_latin.csv", index=False)


# build_lr("scopus-asia-2020")
# # build_lr("scopus-us-2023")
# build_lr("scopus-asia-2023")
# build_lr("scopus-asia-early")
# build_lr("scopus-asia-2017")
# build_lr("scopus-us-2017")

with open("../exp-music/latin_ratio/2020.json", "r") as f:
    musiclr = json.load(f)
with open("../exp-music/latin_ratio/2020-rap-song.json", "r") as f:
    raplr = json.load(f)

music = pd.DataFrame()
music["Latin Ratio"] = musiclr
rap = pd.DataFrame()
rap["Latin Ratio"] = raplr

# dfusearly = pd.DataFrame(pd.read_csv('scopus-early_latin.csv'))
# dfasiaearly = pd.DataFrame(pd.read_csv('scopus-asia-early_latin.csv'))
dfus2017 = pd.DataFrame(pd.read_csv("scopus-us-2017_latin.csv"))
dfasia2017 = pd.DataFrame(pd.read_csv("scopus-asia-2017_latin.csv"))
dfus2020 = pd.DataFrame(pd.read_csv("scopus-us-2020_latin.csv"))
dfus2023 = pd.DataFrame(pd.read_csv("scopus-us-2023_latin.csv"))
dfasia2020 = pd.DataFrame(pd.read_csv("scopus-asia-2020_latin.csv"))
dfasia2023 = pd.DataFrame(pd.read_csv("scopus-asia-2023_latin.csv"))

# dfusearly = dfusearly.loc[dfusearly["Latin Ratio"] != 1]
# dfasiaearly = dfasiaearly.loc[dfasiaearly["Latin Ratio"] != 1]
dfus2017 = dfus2017.loc[dfus2017["Latin Ratio"] != 1]
dfasia2017 = dfasia2017.loc[dfasia2017["Latin Ratio"] != 1]
dfus2020 = dfus2020.loc[dfus2020["Year"] == 2020]
dfus2020 = dfus2020.loc[dfus2020["Latin Ratio"] != 1]
dfasia2020 = dfasia2020.loc[dfasia2020["Latin Ratio"] != 1]
dfus2023 = dfus2023.loc[dfus2023["Latin Ratio"] != 1]
dfasia2023 = dfasia2023.loc[dfasia2023["Latin Ratio"] != 1]

sep = 5

dfus2017 = dfus2017.loc[dfus2017["Cited by"] > sep]
dfus2020 = dfus2020.loc[dfus2020["Cited by"] > sep]
dfus2023 = dfus2023.loc[dfus2023["Cited by"] > sep]
dfasia2017 = dfasia2017.loc[dfasia2017["Cited by"] > sep]
dfasia2020 = dfasia2020.loc[dfasia2020["Cited by"] > sep]
dfasia2023 = dfasia2023.loc[dfasia2023["Cited by"] > sep]


# dfus2017["Citedbyq"] = (dfus2017["Cited by"] < sep).map({True: f"Less than {sep}", False: f"More than {sep}"})
# dfus2020["Citedbyq"] = (dfus2020["Cited by"] < sep).map({True: f"Less than {sep}", False: f"More than {sep}"})
# dfus2023["Citedbyq"] = (dfus2023["Cited by"] < sep).map({True: f"Less than {sep}", False: f"More than {sep}"})
# dfasia2017["Citedbyq"] = (dfasia2017["Cited by"] < sep).map({True: f"Less than {sep}", False: f"More than {sep}"})
# dfasia2020["Citedbyq"] = (dfasia2020["Cited by"] < sep).map({True: f"Less than {sep}", False: f"More than {sep}"})
# dfasia2023["Citedbyq"] = (dfasia2023["Cited by"] < sep).map({True: f"Less than {sep}", False: f"More than {sep}"})

print(len(dfus2017))
# print(len(dfusearly))
print(len(dfus2020))
print(len(dfus2023))
# print(len(dfasiaearly))
print(len(dfasia2017))
print(len(dfasia2020))
print(len(dfasia2023))

# dfusearly['Region'] = 'US'
# dfasiaearly['Region'] = 'Asia'
dfus2017["Region"] = "US"
dfasia2017["Region"] = "Asia"
dfus2020["Region"] = "US"
dfasia2020["Region"] = "Asia"
dfus2023["Region"] = "US"
dfasia2023["Region"] = "Asia"
# dfusearly['Year'] = 2000
# dfasiaearly['Year'] = 2000

abstract = pd.DataFrame()
abstract["Latin Ratio"] = dfus2020["Latin Ratio"]
print(np.mean(music["Latin Ratio"]))
print(np.std(music["Latin Ratio"]))
print(np.mean(abstract["Latin Ratio"]))
print(np.std(abstract["Latin Ratio"]))
print("rap vs music")
print(stats.ttest_ind(rap, music))
print("music vs abstract")
print(stats.ttest_ind(music, abstract))

music["Type"] = "Music"
abstract["Type"] = "Abstract"
musicandabstract = pd.concat([music, abstract])

fig = plt.figure(figsize=(6, 6))
sns.set_theme(context="paper", style="white", palette="pastel")
sns.violinplot(data=musicandabstract, x="Type", y="Latin Ratio")
plt.ylabel("Latin word density")
plt.savefig("music_vs_abstract.pdf")
plt.close()

df = pd.concat([dfus2017, dfus2020, dfus2023, dfasia2017, dfasia2020, dfasia2023])
df = df.drop(columns=["Cited by"])

# descriptive statistics
print(df.groupby(["Region", "Year"]).describe())

df.rename(columns={"Latin Ratio": "LatinRatio"}, inplace=True)
# model = ols("LatinRatio ~ Year + Region + Year * Region", data=df).fit()
# print(sm.stats.anova_lm(model, typ=2))

model = ols("LatinRatio ~ C(Year) * C(Region)", data=df).fit()
print(sm.stats.anova_lm(model, typ=2))

for value in [2017, 2020, 2023]:
    print(f"\nSimple main effect analysis for year={value}")
    sub_df = df[df["Year"] == value]
    model = ols("LatinRatio ~ C(Region)", data=sub_df).fit()
    simple_effect = sm.stats.anova_lm(model, typ=2)
    print(simple_effect)

for value in ["US", "Asia"]:
    print(f"\nSimple main effect analysis for region={value}")
    sub_df = df[df["Region"] == value]
    model = ols("LatinRatio ~ C(Year)", data=sub_df).fit()
    simple_effect = sm.stats.anova_lm(model, typ=2)
    print(simple_effect)
dfasia = df[df["Region"] == "Asia"]
posthoc = pairwise_tukeyhsd(
    endog=dfasia["LatinRatio"], groups=dfasia["Year"], alpha=0.05
)
print(posthoc)

p_adjusted = multipletests(model.pvalues, method="bonferroni")
print(p_adjusted)

fig = plt.figure(figsize=(6, 6))
sns.barplot(data=df, x="Year", y="LatinRatio", hue="Region", capsize=0.1)
plt.ylim(0.4, 0.5)
plt.ylabel("Latin word density")
plt.savefig("us_vs_asia_2017_vs_2020_vs_2023.pdf")

fig = plt.figure(figsize=(6, 6))
sns.barplot(data=df, x="Year", y="LatinRatio", capsize=0.1)
plt.ylim(0.4, 0.5)
plt.ylabel("Latin word density")
plt.savefig("2017_vs_2020_vs_2023.pdf")

fig = plt.figure(figsize=(6, 6))
sns.barplot(data=df, x="Region", y="LatinRatio", capsize=0.1)
plt.ylim(0.4, 0.5)
plt.ylabel("Latin word density")
plt.savefig("us_vs_asia.pdf")

print("asia 2017 vs 2020")
print(stats.ttest_ind(dfasia2017["Latin Ratio"], dfasia2020["Latin Ratio"]))
print("asia 2020 vs 2023")
print(stats.ttest_ind(dfasia2020["Latin Ratio"], dfasia2023["Latin Ratio"]))
print("2017 us vs asia")
print(stats.ttest_ind(dfus2017["Latin Ratio"], dfasia2017["Latin Ratio"]))
print("2020 us vs asia")
print(stats.ttest_ind(dfus2020["Latin Ratio"], dfasia2020["Latin Ratio"]))
print("2023 us vs asia")
print(stats.ttest_ind(dfus2023["Latin Ratio"], dfasia2023["Latin Ratio"]))

# normalize cited by
dfus2017["Cited by n"] = (
    dfus2017["Cited by"] - dfus2017["Cited by"].mean()
) / dfus2017["Cited by"].std()
dfus2020["Cited by n"] = (
    dfus2020["Cited by"] - dfus2020["Cited by"].mean()
) / dfus2020["Cited by"].std()
dfus2023["Cited by n"] = (
    dfus2023["Cited by"] - dfus2023["Cited by"].mean()
) / dfus2023["Cited by"].std()
dfasia2017["Cited by n"] = (
    dfasia2017["Cited by"] - dfasia2017["Cited by"].mean()
) / dfasia2017["Cited by"].std()
dfasia2020["Cited by n"] = (
    dfasia2020["Cited by"] - dfasia2020["Cited by"].mean()
) / dfasia2020["Cited by"].std()
dfasia2023["Cited by n"] = (
    dfasia2023["Cited by"] - dfasia2023["Cited by"].mean()
) / dfasia2023["Cited by"].std()
df["Cited by n"] = (df["Cited by"] - df["Cited by"].mean()) / df["Cited by"].std()

print(stats.pearsonr(df["Cited by n"], df["LatinRatio"]))
print("us 2017")
print(stats.pearsonr(dfus2017["Cited by n"], dfus2017["Latin Ratio"]))
print("us 2020")
print(stats.pearsonr(dfus2020["Cited by n"], dfus2020["Latin Ratio"]))
print("us 2023")
print(stats.pearsonr(dfus2023["Cited by n"], dfus2023["Latin Ratio"]))
print("asia 2017")
print(stats.pearsonr(dfasia2017["Cited by n"], dfasia2017["Latin Ratio"]))
print("asia 2020")
print(stats.pearsonr(dfasia2020["Cited by n"], dfasia2020["Latin Ratio"]))
print("asia 2023")
print(stats.pearsonr(dfasia2023["Cited by n"], dfasia2023["Latin Ratio"]))

fig = plt.figure(figsize=(6, 6))
sns.scatterplot(data=df, y="Cited by n", x="LatinRatio", hue="Region", alpha=0.5)
plt.ylim(min(df["Cited by n"]), 2)
plt.savefig("latin_ratio_citedby.pdf")

fig = plt.figure(figsize=(6, 6))
sns.barplot(data=df, x="Region", y="Cited by", capsize=0.1)
plt.savefig("citedby_us_vs_asia.pdf")

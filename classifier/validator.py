import pandas as pd
import glob
import sys

path = "data/CulturalDeepfake"

files = glob.glob(path+"/human_annotation/*.csv", )



#create an empty set
distinct_values =  set()

labeled_rows = pd.DataFrame(columns=['prediction','validation'])
labeled_df = pd.DataFrame(columns=['Content','validation'])
for file in files:
    # print(file)
    df = pd.read_csv(file)
    #print(df.head())
    df = df[:40]
    # print(df.columns)
    labeled_df = pd.concat([labeled_df, df[['Content','validation']]], axis=0, ignore_index=True)
    labeled_rows = pd.concat([labeled_rows, df[['stance','validation']]], ignore_index=True)
    distinct_values.update(df['validation'].unique())
    # print rows where the validation column is nan
    # print(distinct_values)
    nan_rows = df[df['validation'].isna()]
    # print(nan_rows)

labeled_df.columns = ['text','stance']
labeled_df.to_csv("data/CulturalDeepfake/human_annotation/combined_labeled_rows.csv", index=False)


# generate confusion matrix from labeled rows
confusion_matrix = pd.crosstab(labeled_rows['stance'], labeled_rows['validation'], rownames=['Predicted'], colnames=['Actual'], dropna=False)
print(confusion_matrix)
#compute false negative rate and false positive rate
false_negative_rate = confusion_matrix.loc['believes the fake news', 'criticizes the fake news'] / (confusion_matrix.loc['believes the fake news', 'criticizes the fake news'] + confusion_matrix.loc['believes the fake news', 'believes the fake news'])
false_positive_rate = confusion_matrix.loc['criticizes the fake news', 'believes the fake news'] / (confusion_matrix.loc['criticizes the fake news', 'believes the fake news'] + confusion_matrix.loc['criticizes the fake news', 'criticizes the fake news'])
print("False Negative Rate: ", false_negative_rate)
print("False Positive Rate: ", false_positive_rate)
# compute accuracy
accuracy = (confusion_matrix.loc['believes the fake news', 'believes the fake news'] + confusion_matrix.loc['criticizes the fake news', 'criticizes the fake news']) / confusion_matrix.values.sum()
print("Accuracy: ", accuracy)
# compute precision
precision = confusion_matrix.loc['believes the fake news', 'believes the fake news'] / (confusion_matrix.loc['believes the fake news', 'believes the fake news'] + confusion_matrix.loc['criticizes the fake news', 'believes the fake news'])
print("Precision: ", precision)
# compute recall
recall = confusion_matrix.loc['believes the fake news', 'believes the fake news'] / (confusion_matrix.loc['believes the fake news', 'believes the fake news'] + confusion_matrix.loc['believes the fake news', 'criticizes the fake news'])
print("Recall: ", recall)
# compute f1 score
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1 Score: ", f1_score)
# create confusion matrix heatmap
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(confusion_matrix, annot=True, fmt="d")
plt.savefig("confusion_matrix.png")# generate confusion matrix from labeled rows

def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words and len(word) > 2])
    return text


true_df = pd.read_csv("C:\\D\\FakeNews\\True.csv")
fake_df = pd.read_csv("C:\\D\\FakeNews\\Fake.csv")

true_df["label"] = 1
fake_df["label"] = 0

true_df.drop(columns=["subject", "date"], inplace=True, errors="ignore")
fake_df.drop(columns=["subject", "date"], inplace=True, errors="ignore")

isot_df = pd.concat([true_df, fake_df], ignore_index=True)

welfake_df = pd.read_csv("C:\\D\\FakeNews\\WELFake_Dataset.csv")

welfake_df.drop(columns=["Unnamed: 0"], inplace=True, errors="ignore")

welfake_df["title"].fillna("No Title", inplace=True)

welfake_df.dropna(subset=["text"], inplace=True)

df = pd.concat([isot_df, welfake_df], ignore_index=True)

df["clean_text"] = df["text"].apply(clean_text)

train_df, temp_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"]
)

train_df.to_csv("C:\\E\\Project\\fake_news\\Model_training\\train.csv", index=False)
val_df.to_csv("C:\\E\\Project\\fake_news\\Model_training\\val.csv", index=False)
test_df.to_csv("C:\\E\\Project\\fake_news\\Model_training\\test.csv", index=False)

true_path = "C:/D/FakeNews/True.csv"
fake_path = "C:/D/FakeNews/Fake.csv"
welfake_path = "C:/D/FakeNews/WELFake_Dataset.csv"
output_path = "C:/E/Project/fake_news/data.pkl"


def clean_text(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()


true_df = pd.read_csv(true_path)[["title", "text"]].dropna()
fake_df = pd.read_csv(fake_path)[["title", "text"]].dropna()
welfake_df = pd.read_csv(welfake_path)[["title", "text", "label"]].dropna()

true_df["label"] = 1
fake_df["label"] = 0

true_df["text"] = true_df["text"].apply(clean_text)
fake_df["text"] = fake_df["text"].apply(clean_text)
welfake_df["text"] = welfake_df["text"].apply(clean_text)

data = pd.concat([true_df, fake_df, welfake_df], ignore_index=True)

X = data["text"]
y = data["label"]

vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

pickle.dump((X_train, y_train, X_test, y_test, vectorizer), open(output_path, "wb"))

print("Preprocessing complete. Data saved to:", output_path)

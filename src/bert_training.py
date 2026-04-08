def main():
    train_path = "C:\\E\\Project\\fake_news\\Model_training\\train.csv"
    val_path = "C:\\E\\Project\\fake_news\\Model_training\\val.csv"
    test_path = "C:\\E\\Project\\fake_news\\Model_training\\test.csv"

    train_df = pd.read_csv(train_path).dropna(subset=["clean_text", "label"])
    val_df = pd.read_csv(val_path).dropna(subset=["clean_text", "label"])
    test_df = pd.read_csv(test_path).dropna(subset=["clean_text", "label"])

    PRE_TRAINED_MODEL = "bert-base-uncased"
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL)

    train_dataset = NewsDataset(train_df["clean_text"].to_numpy(), train_df["label"].to_numpy(), tokenizer, MAX_LEN)
    val_dataset = NewsDataset(val_df["clean_text"].to_numpy(), val_df["label"].to_numpy(), tokenizer, MAX_LEN)
    test_dataset = NewsDataset(test_df["clean_text"].to_numpy(), test_df["label"].to_numpy(), tokenizer, MAX_LEN)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = BertForSequenceClassification.from_pretrained(PRE_TRAINED_MODEL, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    history = {"train_loss": [], "val_accuracy": [], "val_f1": []}




def eval_model(model, data_loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    return accuracy, f1, true_labels, preds


def plot_training_history(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.plot(epochs, history["val_f1"], label="Validation F1-score", marker="s")
    plt.xlabel("Epochs")
    plt.ylabel("Score")

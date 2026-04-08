@app.route("/prediction")
def prediction_page():
    return render_template("prediction.html")


@app.route("/detect", methods=["POST"])
def detect_text():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        print("Received Text for Analysis:", text)

        # Predictions
        bert_result = predict_bert(text)
        print("BERT Prediction:", bert_result)

        xgboost_result = predict_xgboost(text)
        print("XGBoost Prediction:", xgboost_result)

        return jsonify({"bert_result": bert_result, "xgboost_result": xgboost_result})

    except Exception as e:
        print("Server Error in /detect:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        gc.collect()


@app.route("/detect_image", methods=["POST"])
def detect_image():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        image_file = request.files["image"]
        img = Image.open(image_file.stream)

        extracted_text = pytesseract.image_to_string(img).strip()

        print("Extracted Text:", extracted_text)

        if not extracted_text:
            return jsonify({"error": "No text extracted from image."}), 400

        bert_result = predict_bert(extracted_text)
        xgboost_result = predict_xgboost(extracted_text)

        return jsonify({
            "extracted_text": extracted_text,
            "bert_result": bert_result,
            "xgboost_result": xgboost_result
        })

    except Exception as e:
        print("Server Error in /detect_image:", e)
        return jsonify({"error": str(e)}), 500

    finally:
        gc.collect()


from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Load DeepSeek model
model_id = "deepseek-ai/deepseek-coder-1.3b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16, device_map="auto"
)

# NL → SQL route
@app.route('/execute_query', methods=['POST'])
def execute_query():
    data = request.json
    nl_query = data.get('query')

    if not nl_query:
        return jsonify({"error": "Query is required!"}), 400

    try:
        prompt = f"""### Instruction:
Convert the following natural language question into a SQL query.

### Input:
{nl_query}

### Output:
"""
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.3,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean SQL output
        sql_start = result.find("### Output:")
        if sql_start != -1:
            sql = result[sql_start + len("### Output:"):].strip()
        else:
            sql = result.strip()

        # Grab first SQL-looking line
        for line in sql.splitlines():
            if line.strip().lower().startswith("select"):
                sql = line.strip()
                break

        return jsonify({
            "success": True,
            "nl_query": nl_query,
            "sql_query": sql
        }), 200

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=10000)

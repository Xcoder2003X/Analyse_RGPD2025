from flask import Flask, request, jsonify
from rag_core.ragcore import RAGSystem
import os
import time

app = Flask(__name__)

# Configuration du préchargement
print("[🔄] Initialisation du système RAG...")
start_time = time.time()

try:
    rag_system = RAGSystem()
    load_time = time.time() - start_time
    print(f"[✅] Système RAG prêt en {load_time:.2f}s!")
except Exception as e:
    print(f"[❌] Échec de l'initialisation: {str(e)}")
    raise SystemExit(1)

@app.route('/query', methods=['POST'])
def handle_query():
    try:
        data = request.json
        response = rag_system.query(
            data.get('question', ''),
            top_k=data.get('top_k', 3)
        )
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": "RAG_ERROR"
        }), 500

if __name__ == '__main__':
    from waitress import serve
    print("[🚀] Démarrage du serveur en mode production...")
    serve(app, host='0.0.0.0', port=5000)
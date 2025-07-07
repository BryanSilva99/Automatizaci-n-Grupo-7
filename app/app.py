from flask import Flask, render_template, request, jsonify
from grupo7 import cluster_repositories_with_genetic_algorithm, draw_topology_interactive

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/cluster', methods=['POST'])
def cluster():
    try:
        data = request.get_json()
        repos_raw = data.get('repos', '')
        repos = [r.strip() for r in repos_raw.split('\n') if r.strip()]

        # Ejecuta el análisis de clústeres
        clusters, explanations = cluster_repositories_with_genetic_algorithm(
            repos, num_clusters=3)

        # Genera imagen de topología (solo del primer repo por simplicidad)
        topology_image = None
        if repos:
            # importa internamente si no está arriba
            from grupo7 import parse_docker_compose_or_yaml
            repo_url = repos[0]
            # asegúrate de importar esto también
            from grupo7 import clone_repo, cleanup_repo
            temp_dir = clone_repo(repo_url)
            if temp_dir:
                topology = parse_docker_compose_or_yaml(temp_dir)
                topology_image = draw_topology_interactive(
                    topology, repo_url=repo_url)
                cleanup_repo(temp_dir)

        return jsonify({
            'clusters': clusters,
            'explanations': explanations,
            'topology_image': topology_image
        })

    except Exception as e:
        print("🔥 ERROR INTERNO:", e)  # 👈 Agrega esta línea
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

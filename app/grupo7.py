

import os
from dotenv import load_dotenv
import stat
import shutil
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
from matplotlib.patches import Patch
from collections import defaultdict
import matplotlib.pyplot as plt
import yaml
import git
import tempfile
import requests
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # 🔥 Fuerza matplotlib a usar un backend sin GUI


# ✅ Calcula la ruta absoluta al archivo .env que está en la raíz del proyecto
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(base_dir, ".env")

# ✅ Carga las variables del archivo .env
load_dotenv(dotenv_path)

# ✅ Recupera las variables
groq_api_key = os.getenv("GROQ_API_KEY")
groq_url = os.getenv("GROQ_API_URL")

if not groq_api_key or not groq_url:
    raise ValueError(
        "❌ No se encontró GROQ_API_KEY o GROQ_API_URL en el entorno")


def obtener_respuesta_groq(mensaje):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {groq_api_key}"
    }
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [{
            "role": "user",
            "content": mensaje
        }]
    }
    try:
        response = requests.post(groq_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "No se obtuvo respuesta.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error al hacer la solicitud a la API de Groq: {e}")
        return None


# Tecnologías que buscas en el repositorio
FEATURES = ["Docker", "Kubernetes", "CI/CD", "Helm", "Terraform", "Python", "Node.js", "Java", "Ruby", "Go",
            "PostgreSQL", "MySQL", "Redis", "MongoDB", "Nginx", "Apache", "React", "Vue", "Angular", "FastAPI"]


def clone_repo(repo_url):
    """Clona un repositorio en un directorio temporal"""
    try:
        temp_dir = tempfile.mkdtemp()
        print(f"Clonando {repo_url} en {temp_dir}")
        git.Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except Exception as e:
        print(f"Error al clonar {repo_url}: {e}")
        return None


def handle_remove_readonly(func, path, exc_info):
    import os
    os.chmod(path, stat.S_IWRITE)
    func(path)


def cleanup_repo(path):
    try:
        shutil.rmtree(path, onerror=handle_remove_readonly)
    except Exception as e:
        print("⚠️ Error al eliminar el repositorio clonado:", e)


def parse_yaml_files(repo_path):
    """Analiza archivos YAML en el repositorio, manejando múltiples documentos."""
    yaml_files = []

    # Buscar archivos .yaml y .yml
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                file_path = os.path.join(root, file)
                yaml_files.append(file_path)

    # Parsear cada archivo YAML encontrado
    all_yaml_docs = []
    yaml_content = ""

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                yaml_content += content + "\n"

                # Permitir múltiples documentos en un solo archivo YAML
                try:
                    documents = list(yaml.safe_load_all(content))
                    all_yaml_docs.extend(
                        [doc for doc in documents if doc is not None])
                except yaml.YAMLError:
                    # Si falla el parsing YAML, al menos tenemos el contenido como texto
                    pass
        except Exception as e:
            print(f"Error al leer el archivo YAML {yaml_file}: {e}")

    return all_yaml_docs, yaml_content


def parse_docker_compose_or_yaml(repo_path):
    """Analiza docker-compose.yml o cualquier archivo .yaml en busca de dependencias"""
    docker_compose_files = []
    docker_compose_dependencies = []

    # Buscar archivos docker-compose.yml y otros YAMLs
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if "docker-compose" in file.lower() and (file.endswith(".yml") or file.endswith(".yaml")):
                docker_compose_files.append(os.path.join(root, file))

    # Analizar todos los archivos docker-compose
    for docker_compose_file in docker_compose_files:
        try:
            with open(docker_compose_file, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()

                # Permitir múltiples documentos en un solo archivo YAML
                try:
                    documents = list(yaml.safe_load_all(content))

                    for doc in documents:
                        if doc and "services" in doc:
                            for service, data in doc["services"].items():
                                dependencies = []

                                # Buscar diferentes tipos de dependencias
                                if "depends_on" in data:
                                    if isinstance(data["depends_on"], list):
                                        dependencies.extend(data["depends_on"])
                                    elif isinstance(data["depends_on"], dict):
                                        dependencies.extend(
                                            data["depends_on"].keys())

                                # Buscar links
                                if "links" in data:
                                    dependencies.extend(data["links"])

                                docker_compose_dependencies.append({
                                    "name": service,
                                    "dependencies": dependencies,
                                    "image": data.get("image", ""),
                                    "ports": data.get("ports", []),
                                    "environment": data.get("environment", {})
                                })
                except yaml.YAMLError as e:
                    print(
                        f"Error al parsear YAML en {docker_compose_file}: {e}")

        except Exception as e:
            print(f"Error al analizar el archivo {docker_compose_file}: {e}")

    return docker_compose_dependencies


def extract_readme_or_yaml(repo_path):
    """Extrae el contenido del README.md y de archivos YAML para análisis"""
    content = ""

    # Extraer README
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.lower() in ["readme.md", "readme.txt", "readme.rst"]:
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content += f.read() + "\n"
                except Exception as e:
                    print(f"Error al leer {file}: {e}")

    # Extraer contenido de archivos YAML
    _, yaml_content = parse_yaml_files(repo_path)
    content += yaml_content

    # Extraer contenido de Dockerfile
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.lower() == "dockerfile" or file.lower().startswith("dockerfile"):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                        content += f.read() + "\n"
                except Exception as e:
                    print(f"Error al leer {file}: {e}")

    return content


def extract_technologies_from_text(text):
    """Extrae las tecnologías del texto y de estructuras YAML"""
    technologies = set()
    text_lower = text.lower()

    # Buscar tecnologías por palabras clave
    for feature in FEATURES:
        if feature.lower() in text_lower:
            technologies.add(feature)

    # Buscar patrones específicos
    patterns = {
        "Docker": ["dockerfile", "docker-compose", "FROM ", "RUN ", "COPY ", "ADD "],
        "Kubernetes": ["apiVersion:", "kind:", "metadata:", "spec:", "kubectl"],
        "PostgreSQL": ["postgres", "postgresql", "psql"],
        "MySQL": ["mysql", "mariadb"],
        "Redis": ["redis"],
        "MongoDB": ["mongo", "mongodb"],
        "Nginx": ["nginx"],
        "Apache": ["apache", "httpd"],
        "Python": [".py", "python", "pip", "requirements.txt", "setup.py"],
        "Node.js": ["package.json", "npm", "node", "yarn"],
        "Java": [".java", "maven", "gradle", "pom.xml"],
        "Go": [".go", "go.mod", "go.sum"],
        "React": ["react", "jsx"],
        "Vue": ["vue"],
        "Angular": ["angular", "@angular"],
        "FastAPI": ["fastapi", "uvicorn"]
    }

    for tech, keywords in patterns.items():
        for keyword in keywords:
            if keyword in text_lower:
                technologies.add(tech)
                break

    return list(technologies)


def generate_topology_explanation(topology):
    """Genera una explicación breve de la topología usando Groq"""
    if not topology:
        return "No se encontraron componentes para analizar."

    services = [s['name'] for s in topology]
    dependencies_info = []

    for service in topology:
        if service.get('dependencies'):
            deps = ', '.join(service['dependencies'])
            dependencies_info.append(f"{service['name']} depende de: {deps}")

    mensaje = f"""Analiza brevemente esta arquitectura de microservicios en máximo 3 líneas (no menciones que es en 3 líneas):
Servicios: {', '.join(services)}
Dependencias: {'; '.join(dependencies_info) if dependencies_info else 'Sin dependencias explícitas'}
Describe la relación entre componentes de forma concisa."""

    explicacion = obtener_respuesta_groq(mensaje)
    return explicacion if explicacion else "Arquitectura con servicios interconectados según dependencias definidas."


def draw_topology_interactive(topology, title="Mapa Topológico", repo_url="", image_name="topo.png"):
    if not topology:
        print(f"⚠️ No se encontró topología para {repo_url}")
        return None

    G = nx.DiGraph()
    for service in topology:
        G.add_node(service["name"])
        for dep in service.get("dependencies", []):
            G.add_edge(service["name"], dep)

    if len(G.nodes()) == 0:
        print(f"⚠️ No se encontraron servicios en {repo_url}")
        return None

    # Crear grafo
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue',
                           node_size=3000, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edge_color='gray',
                           arrows=True, arrowsize=20, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    plt.title(f"{title}\n{repo_url}")
    plt.axis('off')
    plt.tight_layout()

    # ✅ Obtener ruta absoluta segura al directorio static/
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    os.makedirs(static_dir, exist_ok=True)

    output_path = os.path.join(static_dir, image_name)
    plt.savefig(output_path)
    plt.close()

    return f"/static/{image_name}"


def genetic_algorithm(features, num_clusters=3):
    """Algoritmo genético simple para agrupación"""
    if not features:
        return []

    population_size = 50
    generations = 100
    mutation_rate = 0.1

    # Inicializar población
    population = []
    for _ in range(population_size):
        individual = [random.randint(0, num_clusters - 1)
                      for _ in range(len(features))]
        population.append(individual)

    def fitness(individual):
        # Calcular fitness basado en cohesión intra-cluster y separación inter-cluster
        clusters = defaultdict(list)
        for i, cluster_id in enumerate(individual):
            clusters[cluster_id].append(features[i])

        total_fitness = 0
        for cluster_features in clusters.values():
            if len(cluster_features) > 1:
                # Calcular similitud promedio dentro del cluster
                similarities = []
                for i in range(len(cluster_features)):
                    for j in range(i + 1, len(cluster_features)):
                        sim = cosine_similarity([cluster_features[i]], [
                                                cluster_features[j]])[0][0]
                        similarities.append(sim)
                if similarities:
                    total_fitness += np.mean(similarities)

        return total_fitness

    # Evolución
    for generation in range(generations):
        # Evaluar fitness
        fitness_scores = [fitness(ind) for ind in population]

        # Selección
        sorted_population = [x for _, x in sorted(
            zip(fitness_scores, population), reverse=True)]

        # Nueva generación
        # Mantener los mejores
        new_population = sorted_population[:population_size // 2]

        # Crossover y mutación
        while len(new_population) < population_size:
            parent1 = random.choice(sorted_population[:10])
            parent2 = random.choice(sorted_population[:10])

            # Crossover
            crossover_point = random.randint(1, len(features) - 1)
            child = parent1[:crossover_point] + parent2[crossover_point:]

            # Mutación
            for i in range(len(child)):
                if random.random() < mutation_rate:
                    child[i] = random.randint(0, num_clusters - 1)

            new_population.append(child)

        population = new_population

    # Retornar el mejor individuo
    fitness_scores = [fitness(ind) for ind in population]
    best_individual = population[fitness_scores.index(max(fitness_scores))]

    return best_individual


def generate_cluster_explanation(clusters, repo_technologies):
    """Genera explicación de clusters usando Groq"""
    cluster_descriptions = []

    for cluster_id, repos in clusters.items():
        if not repos:
            continue

        # Obtener tecnologías comunes del cluster
        cluster_tech = []
        for repo in repos:
            if repo in repo_technologies:
                cluster_tech.extend(repo_technologies[repo])

        # Contar frecuencias de tecnologías
        tech_counts = defaultdict(int)
        for tech in cluster_tech:
            tech_counts[tech] += 1

        common_techs = [tech for tech,
                        count in tech_counts.items() if count > 1]

        mensaje = f"""Explica de manera muy breve (por ejemplo, un par de líneas) por qué estos repositorios están agrupados:
Repositorios: {', '.join([r.split('/')[-1] for r in repos])}
Tecnologías comunes: {', '.join(common_techs) if common_techs else 'Variadas'}
Razón de agrupación:"""

        explicacion = obtener_respuesta_groq(mensaje)
        if explicacion:
            cluster_descriptions.append(f"Clúster {cluster_id}: {explicacion}")
        else:
            cluster_descriptions.append(
                f"Clúster {cluster_id}: Agrupados por similitud tecnológica y arquitectónica.")

    return cluster_descriptions


def cluster_repositories_with_genetic_algorithm(repos, num_clusters=3):
    repo_features = []
    repo_explanations = []
    successful_repos = []
    repo_technologies = {}  # Para almacenar las tecnologías de cada repo

    # Extraer características y explicaciones
    for repo_url in repos:
        print(f"\n--- Analizando {repo_url} ---")
        repo_path = clone_repo(repo_url)

        if repo_path is None:
            print(f"Saltando {repo_url} debido a error en clonación")
            continue

        try:
            # Extraer texto de README y archivos YAML
            text = extract_readme_or_yaml(repo_path)
            technologies = extract_technologies_from_text(text)
            print(f"Tecnologías encontradas: {technologies}")

            # Codificar características
            encoded_features = [
                1 if f in technologies else 0 for f in FEATURES]
            repo_features.append(encoded_features)
            successful_repos.append(repo_url)
            repo_technologies[repo_url] = technologies  # Almacenar tecnologías

            explanation = f"Repositorio: {repo_url} usa las tecnologías: {', '.join(technologies) if technologies else 'Ninguna detectada'}"
            repo_explanations.append(explanation)

            # Generar y mostrar mapa topológico
            topology = parse_docker_compose_or_yaml(repo_path)
            if topology:
                print(
                    f"Servicios encontrados: {[s['name'] for s in topology]}")
                draw_topology_interactive(
                    topology, f"Mapa Topológico", repo_url)
            else:
                print("No se encontró archivo .yml o servicios")

        except Exception as e:
            print(f"Error procesando {repo_url}: {e}")
        finally:
            # Limpiar directorio temporal
            cleanup_repo(repo_path)

    if not repo_features:
        print("No se pudieron procesar repositorios")
        return {}, []

    # Ejecutar algoritmo genético para agrupación
    print(
        f"\nEjecutando algoritmo genético con {len(successful_repos)} repositorios...")
    best_individual = genetic_algorithm(repo_features, num_clusters)

    # Agrupación final
    clusters = defaultdict(list)
    for repo_idx, cluster_id in enumerate(best_individual):
        clusters[cluster_id].append(successful_repos[repo_idx])

    # Generar explicaciones de clusters con Groq
    cluster_explanations = generate_cluster_explanation(
        clusters, repo_technologies)

    return clusters, cluster_explanations


# MAIN
if __name__ == "__main__":
    repos = [
        "https://github.com/microservices-demo/microservices-demo",
        "https://github.com/spring-petclinic/spring-petclinic-microservices",
        "https://github.com/kgrzybek/modular-monolith-with-ddd",
        "https://github.com/tiangolo/full-stack-fastapi-postgresql",
        "https://github.com/react-boilerplate/react-boilerplate",
        "https://github.com/kelseyhightower/kubernetes-the-hard-way",
        "https://github.com/eventuate-tram/eventuate-tram-examples-customers-and-orders",
        "https://github.com/didtheyghostme/didtheyghostme",
        "https://github.com/palinkiewicz/lyricpost",
        "https://github.com/petrousoft/kubejyg"
    ]

    print("Iniciando análisis de repositorios...")
    clusters, explanations = cluster_repositories_with_genetic_algorithm(
        repos, num_clusters=3)

    print("\n=== RESULTADOS DE CLUSTERING ===")
    for i, cluster in clusters.items():
        print(f"\nClúster {i}:")
        for repo in cluster:
            print(f"  - {repo}")

    print("\n=== EXPLICACIONES DE AGRUPACIÓN ===")
    for explanation in explanations:
        print(f"🔍 {explanation}")

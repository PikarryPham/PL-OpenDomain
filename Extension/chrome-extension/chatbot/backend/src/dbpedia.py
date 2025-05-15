import logging
import requests
import xml.etree.ElementTree as ET

from SPARQLWrapper import SPARQLWrapper, JSON

# Define the SPARQL endpoint
sparql = SPARQLWrapper("http://34.42.208.64:3030/dbpedia/query")

# Set authentication credentials
sparql.setCredentials("admin", "Eww7znESXVQhY0n")

search_url = "https://lookup.dbpedia.org/api/search/KeywordSearch"


def get_data(query):
    # Set the query and return format
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    # Execute the query
    results = sparql.query().convert()
    logging.info("SPARQL results: {}".format(results))

    return results["results"]["bindings"]


def get_page_detail(uri):
    # Convert the page URL to the resource URL
    if uri.startswith("https://dbpedia.org/page/"):
        uri = uri.replace("https://dbpedia.org/page/", "http://dbpedia.org/resource/")
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbp: <http://dbpedia.org/property/>

        SELECT DISTINCT 
            ?property 
            ?value 
        WHERE {{
            # Lấy abstract từ DBpedia
            {{
                <{uri}> dbo:abstract ?value .
                BIND(str("DBpedia Abstract") AS ?property)
            }}
            UNION 
            {{
                # Lấy comment từ RDF Schema
                <{uri}> rdfs:comment ?value .
                BIND(str("RDF Schema Comment") AS ?property)
            }}
            UNION 
            {{
                # Các thuộc tính mô tả khác của DBpedia
                <{uri}> ?p ?value .
                FILTER(
                    CONTAINS(str(?p), "description") || 
                    CONTAINS(str(?p), "abstract") || 
                    CONTAINS(str(?p), "comment") ||
                    ?p = dbp:description
                )
                BIND(str(?p) AS ?property)
            }}

            # Lọc kết quả
            FILTER(
                LANG(?value) = "en" || # Chỉ lấy tiếng Anh
                LANG(?value) = ""      # Hoặc các giá trị không có ngôn ngữ
            )
        }} 
        """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data = {}
    for result in results["results"]["bindings"]:
        property_uri = result["property"]["value"]
        property_value = result["value"]["value"]
        if property_uri == "http://www.w3.org/2000/01/rdf-schema#comment":
            property_uri = "comment"
        if property_uri in data:
            data[property_uri].append(property_value)
        else:
            data[property_uri] = [property_value]

    return data


def get_page_relationship(name):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dbc: <http://dbpedia.org/resource/Category:>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?relatedConcept ?relatedConceptLabel ?relationshipType
        WHERE {{
          {{
            # Phương pháp 1: Sử dụng skos:broader để tìm subcategory
            ?relatedConcept skos:broader {name} .
            BIND("subcategory (skos:broader)" AS ?relationshipType)
          }}
          UNION 
          {{
            # Lấy các trang được liên kết từ Machine Learning
            {name} dbo:wikiPageWikiLink ?relatedConcept .
            BIND("outgoing link" AS ?relationshipType)
          }}
          UNION 
          {{
            # Lấy các trang liên kết đến Machine Learning
            ?relatedConcept dbo:wikiPageWikiLink {name} .
            BIND("incoming link" AS ?relationshipType)
          }}
          
          # Lấy nhãn của các concept liên quan nếu có
          OPTIONAL {{
            ?relatedConcept rdfs:label ?relatedConceptLabel .
            FILTER(LANG(?relatedConceptLabel) = "en")
          }}
          
          # Loại bỏ Machine Learning từ kết quả
          FILTER(?relatedConcept != {name})
        }}
        ORDER BY ?relationshipType ?relatedConceptLabel
        LIMIT 50
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data = {}
    for result in results["results"]["bindings"]:
        property_uri = result["property"]["value"]
        property_value = result["value"]["value"]
        if property_uri == "http://www.w3.org/2000/01/rdf-schema#comment":
            property_uri = "comment"
        if property_uri in data:
            data[property_uri].append(property_value)
        else:
            data[property_uri] = [property_value]

    return data


def get_all_pages_of_category(category_suffix):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dbc: <http://dbpedia.org/resource/Category:>
        PREFIX dbo: <http://dbpedia.org/ontology/>
        PREFIX dbr: <http://dbpedia.org/resource/>
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        
        SELECT DISTINCT ?relatedConcept ?relatedConceptLabel ?relationshipType
        WHERE {{
          {{
            # Phương pháp 1: Sử dụng skos:broader để tìm subcategory
            ?relatedConcept skos:broader dbc:{category_suffix} .
            BIND("subcategory (skos:broader)" AS ?relationshipType)
          }}
          UNION 
          {{
            # Lấy các trang được liên kết từ $category_suffix
            dbr:{category_suffix} dbo:wikiPageWikiLink ?relatedConcept .
            BIND("outgoing link" AS ?relationshipType)
          }}
          UNION 
          {{
            # Lấy các trang liên kết đến $category_suffix
            ?relatedConcept dbo:wikiPageWikiLink dbr:{category_suffix} .
            BIND("incoming link" AS ?relationshipType)
          }}
          
          # Lấy nhãn của các concept liên quan nếu có
          OPTIONAL {{
            ?relatedConcept rdfs:label ?relatedConceptLabel .
            FILTER(LANG(?relatedConceptLabel) = "en")
          }}
          
          # Loại bỏ $category_suffix từ kết quả
          FILTER(?relatedConcept != dbr:{category_suffix})
        }}
        ORDER BY ?relationshipType ?relatedConceptLabel
        LIMIT 50
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data = {}
    for result in results["results"]["bindings"]:
        uri = result["relatedConcept"]["value"] if "relatedConcept" in result else None
        label = (
            result["relatedConceptLabel"]["value"]
            if "relatedConceptLabel" in result
            else None
        )
        if uri and label:
            data[uri] = label

    return data


def get_category_detail(uri):
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")
    query = f"""
        SELECT ?property ?value
        WHERE {{
            {{
                <{uri}> ?property ?value
            }}
            UNION
            # Lấy abstract từ DBpedia
            {{
                <{uri}> dbo:abstract ?value .
                BIND(str("DBpedia Abstract") AS ?property)
            }}
            UNION 
            {{
                # Lấy comment từ RDF Schema
                <{uri}> rdfs:comment ?value .
                BIND(str("RDF Schema Comment") AS ?property)
            }}
            UNION 
            {{
                # Các thuộc tính mô tả khác của DBpedia
                <{uri}> ?p ?value .
                FILTER(
                    CONTAINS(str(?p), "description") || 
                    CONTAINS(str(?p), "abstract") || 
                    CONTAINS(str(?p), "comment") ||
                    ?p = dbp:description
                )
                BIND(str(?p) AS ?property)
            }}
            UNION
            {{
                ?value ?property <{uri}>
                FILTER(?property IN (dbo:wikiPageWikiLink, dcterms:subject))
            }}
        }}
        LIMIT 50
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    data = {}
    logging.info("SPARQL results: {}".format(results))
    for result in results["results"]["bindings"]:
        property_uri = result["property"]["value"]
        property_value = result["value"]["value"]
        if property_uri == "http://dbpedia.org/ontology/wikiPageWikiLink":
            property_uri = "wikiPageWikiLink"

        if property_uri in data:
            data[property_uri].append(property_value)
        else:
            data[property_uri] = [property_value]

    return data


def search_page(keyword):
    params = {"QueryString": keyword}

    headers = {"Accept": "application/json"}
    response = requests.get(search_url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.text
        return data
    else:
        raise AttributeError(f"Request failed: {response.text}")


def parse_response(xml_data, limit=2):
    root = ET.fromstring(xml_data)
    results = {}
    topic_label = None
    topic_uri = None
    search_results = root.findall(".//Result")

    for i in range(len(search_results)):
        result = search_results[i]
        if topic_uri is None and topic_label is None:
            topic_uri = result.find("URI").text
            topic_label = topic_uri.split("/")[-1]

        for category in result.findall(".//Category"):
            uri = category.find("URI").text
            if "resource" in uri:
                label = uri.split("/")[-1].split(":")[-1]
                results[uri] = label

        if i >= limit:
            break

    return topic_uri, topic_label, results


def whitelist_concepts():
    return [
        "Data Mining",
        "Machine Learning",
        "Deep Learning",
        "Natural Language Processing",
        "Computer Vision",
        "Software Development",
        "Data Science",
        "Data Analysis",
        "Applied Statistics",
        "Data Visualization",
        "Feature Engineering",
        "Big Data",
        "Predictive Analytics",
        "Prescriptive Analytics",
        "Real-time Analytics",
        "Data Processing",
        "Data Cleaning",
        "Model Evaluation",
        "Reinforcement Learning",
        "Recommendation Systems",
        "Time Series Analysis",
        "Data Engineering",
        "Data Governance",
        "Data Ethics",
        "Computational Science",
        "Supervised Learning",
        "Unsupervised Learning",
        "Bioinformatics",
        "Optimization",
        "Cloud Computing",
        "Text Analytics",
        "Social Network Analysis",
        "Geospatial Data Science",
        "Distributed Databases",
        "Distributed Data Processing",
        "Online Learning",
        "Transfer Learning",
        "Artificial Intelligence",
        "Machine Learning",
        "Cybersecurity",
        "Cloud Computing",
        "Web Development",
        "Mobile App Development",
        "Internet of Things (IoT)",
        "Hardware Engineering",
        "Embedded Systems",
        "Robotics",
        "Blockchain",
        "Cryptocurrency",
        "Game Development",
        "Augmented Reality",
        "UI/UX Design",
        "Computer Networks",
        "Databases and DBMS",
        "Operating Systems",
        "Computer Graphics",
        "VR/AR",
        "Natural Language Processing",
        "Computer Architecture",
        "Quantum Computing",
        "High-Performance Computing",
        "Big Data",
        "DevOps",
        "Automation",
        "Social Computing",
        "Technical Standards",
        "Data Visualization",
        "Data Ethics",
        "Protocols",
        "Technology Ethics",
        "Business",
        "Finance",
        "Economics",
        "Marketing",
        "Creative Arts",
        "Design",
        "Health",
        "Medicine",
        "Language",
        "Linguistics",
        "Communication",
        "STEM",
        "Science",
        "Technology",
        "Engineering",
        "Philosophy",
        "Mathematics",
        "Personal Development",
        "Professional Development",
        "Education",
        "Social Sciences",
        "Humanities",
        "Environmental Studies",
        "Legal Studies",
        "Entertainment",
        "Media Studies",
        "Sports",
        "Fitness",
        "Culinary Arts",
        "Food Science",
        "Agriculture",
        "Farming",
        "Transportation",
        "Logistics",
        "Manufacturing",
        "Industrial Arts",
        "Military",
        "Defense Studies",
        "Religious",
        "Spirituality",
        "Tourism",
        "Hospitality",
        "Energy",
        "Utilities",
        "Psychology",
        "Mental Health",
        "Cultural Studies",
        "History",
        "Anthropology",
    ]


if __name__ == "__main__":
    search_response = get_category_detail(
        "http://dbpedia.org/resource/Category:Support_vector_machines"
    )
    # parsed = parse_response(search_response)
    print(search_response)

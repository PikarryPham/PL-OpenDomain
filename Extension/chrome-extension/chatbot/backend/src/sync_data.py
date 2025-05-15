from dbpedia import search_page, parse_response, whitelist_concepts
from models import insert_topic, insert_category, insert_page


def collect_dbpedia_topics(list_concept):
    """
    Collect DBpedia resources for a list of concepts.

    Args:
        list_concept (list): A list of concepts.

    Returns:
        dict: A dictionary of DBpedia resources.
    """
    dbpedia_resources = {}

    for concept in list_concept:
        if isinstance(concept, list):
            # Recursively collect resources for nested lists
            results = collect_dbpedia_topics(concept)
            dbpedia_resources.update(results)
        else:
            search_response = search_page(concept)
            topic_uri, topic_label, parsed_results = parse_response(search_response)
            topic_record = insert_topic(topic_label, topic_uri)
            parent_id = topic_record.id
            dbpedia_resources[parent_id] = parsed_results

    return dbpedia_resources

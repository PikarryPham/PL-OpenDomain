import logging
import sys
sys.path.append('/home/ubuntu')

from config import *
from db_connector import Neo4jConnector

# Cấu hình logging hiển thị ra console
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()  # Thêm StreamHandler để hiển thị log ra console
    ]
)
logger = logging.getLogger(__name__)

def run_test_queries():
    """
    Chạy các truy vấn kiểm tra để xác nhận KG đã được xây dựng đúng
    """
    try:
        connector = Neo4jConnector()
        
        # 1. Kiểm tra số lượng node của mỗi loại
        print("\n=== KIỂM TRA SỐ LƯỢNG NODE ===")
        node_types = ["User", "Question", "Option", "Entry", "Topic", "Category", "RelatedConcept"]
        for node_type in node_types:
            query = f"MATCH (n:{node_type}) RETURN count(n) as count"
            result = connector.run_query(query)
            count = result[0]["count"] if result else 0
            print(f"Số lượng node {node_type}: {count}")
        
        # 2. Kiểm tra số lượng relationship của mỗi loại
        print("\n=== KIỂM TRA SỐ LƯỢNG RELATIONSHIP ===")
        rel_types = ["HAS", "CHOOSE", "RELATE_TO", "HAS_ROOT", "HAS_PARENT"]
        for rel_type in rel_types:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count"
            result = connector.run_query(query)
            count = result[0]["count"] if result else 0
            print(f"Số lượng relationship {rel_type}: {count}")
        
        # 3. Kiểm tra mối quan hệ giữa User và Entry
        print("\n=== KIỂM TRA MỐI QUAN HỆ GIỮA USER VÀ ENTRY ===")
        query = """
        MATCH (u:User)-[r:HAS]->(e:Entry)
        RETURN u.user_id as user_id, count(e) as entry_count
        LIMIT 5
        """
        result = connector.run_query(query)
        for record in result:
            print(f"User {record['user_id']} có {record['entry_count']} entry")
        
        # 4. Kiểm tra mối quan hệ giữa User và Option
        print("\n=== KIỂM TRA MỐI QUAN HỆ GIỮA USER VÀ OPTION ===")
        query = """
        MATCH (u:User)-[r:CHOOSE]->(o:Option)
        RETURN u.user_id as user_id, count(o) as option_count
        LIMIT 5
        """
        result = connector.run_query(query)
        for record in result:
            print(f"User {record['user_id']} đã chọn {record['option_count']} option")
        
        # 5. Kiểm tra mối quan hệ giữa Question và Option
        print("\n=== KIỂM TRA MỐI QUAN HỆ GIỮA QUESTION VÀ OPTION ===")
        query = """
        MATCH (q:Question)-[r:HAS]->(o:Option)
        RETURN q.id as question_id, count(o) as option_count
        LIMIT 5
        """
        result = connector.run_query(query)
        for record in result:
            print(f"Question {record['question_id']} có {record['option_count']} option")
        
        # 6. Kiểm tra mối quan hệ giữa Entry và Topic
        print("\n=== KIỂM TRA MỐI QUAN HỆ GIỮA ENTRY VÀ TOPIC ===")
        query = """
        MATCH (e:Entry)-[r:RELATE_TO]->(t:Topic)
        RETURN e.entry_id as entry_id, t.label as topic_label, r.weightView as weight_view, r.weightDurationPerSession as weight_duration
        LIMIT 5
        """
        result = connector.run_query(query)
        for record in result:
            print(f"Entry {record['entry_id']} liên quan đến topic {record['topic_label']} với weightView={record['weight_view']} và weightDuration={record['weight_duration']}")
        
        # 7. Kiểm tra mối quan hệ giữa Category và Topic
        print("\n=== KIỂM TRA MỐI QUAN HỆ GIỮA CATEGORY VÀ TOPIC ===")
        query = """
        MATCH (c:Category)-[r:HAS_ROOT]->(t:Topic)
        RETURN c.label as category_label, t.label as topic_label
        LIMIT 5
        """
        result = connector.run_query(query)
        for record in result:
            print(f"Category {record['category_label']} có root là topic {record['topic_label']}")
        
        # 8. Kiểm tra mối quan hệ giữa RelatedConcept và Category
        print("\n=== KIỂM TRA MỐI QUAN HỆ GIỮA RELATEDCONCEPT VÀ CATEGORY ===")
        query = """
        MATCH (r:RelatedConcept)-[rel:HAS_PARENT]->(c:Category)
        RETURN r.label as concept_label, c.label as category_label
        LIMIT 5
        """
        result = connector.run_query(query)
        for record in result:
            print(f"RelatedConcept {record['concept_label']} có parent là category {record['category_label']}")
        
        # Đóng kết nối
        connector.close()
        
    except Exception as e:
        print(f"Lỗi khi chạy truy vấn kiểm tra: {str(e)}")

if __name__ == "__main__":
    run_test_queries()

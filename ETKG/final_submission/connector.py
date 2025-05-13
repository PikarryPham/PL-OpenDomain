import logging
import sys
from neo4j import GraphDatabase
sys.path.append('/home/ubuntu')
from config import *

logger = logging.getLogger(__name__)

class Neo4jConnector:
    """
    Lớp kết nối với Neo4j
    """
    
    def __init__(self, uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
        """
        Khởi tạo kết nối với Neo4j
        
        Args:
            uri (str): URI của Neo4j
            user (str): Tên người dùng
            password (str): Mật khẩu
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            logger.info("Đã kết nối thành công với Neo4j")
        except Exception as e:
            logger.error(f"Lỗi khi kết nối với Neo4j: {str(e)}")
            raise
    
    def close(self):
        """
        Đóng kết nối với Neo4j
        """
        if self.driver:
            self.driver.close()
            logger.info("Đã đóng kết nối với Neo4j")
    
    def run_query(self, query, parameters=None):
        """
        Thực thi Cypher query
        
        Args:
            query (str): Cypher query
            parameters (dict): Tham số cho query
            
        Returns:
            list: Kết quả của query
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return list(result)
        except Exception as e:
            logger.error(f"Lỗi khi thực thi query: {str(e)}")
            logger.error(f"Query: {query}")
            logger.error(f"Parameters: {parameters}")
            raise
    
    def create_constraints_and_indexes(self):
        """
        Tạo các ràng buộc và chỉ mục cho Neo4j
        """
        # Tạo ràng buộc cho User
        self.run_query("CREATE CONSTRAINT user_id_constraint IF NOT EXISTS FOR (u:User) REQUIRE u.user_id IS UNIQUE")
        
        # Tạo ràng buộc cho Question
        self.run_query("CREATE CONSTRAINT question_id_constraint IF NOT EXISTS FOR (q:Question) REQUIRE q.id IS UNIQUE")
        
        # Tạo ràng buộc cho Option
        self.run_query("CREATE CONSTRAINT option_id_constraint IF NOT EXISTS FOR (o:Option) REQUIRE o.id IS UNIQUE")
        
        # Tạo ràng buộc cho Entry
        self.run_query("CREATE CONSTRAINT entry_id_constraint IF NOT EXISTS FOR (e:Entry) REQUIRE e.entry_id IS UNIQUE")
        
        # Tạo ràng buộc cho Topic
        self.run_query("CREATE CONSTRAINT topic_uri_constraint IF NOT EXISTS FOR (t:Topic) REQUIRE t.uri IS UNIQUE")
        
        # Tạo ràng buộc cho Category
        self.run_query("CREATE CONSTRAINT category_uri_constraint IF NOT EXISTS FOR (c:Category) REQUIRE c.uri IS UNIQUE")
        
        # Tạo ràng buộc cho RelatedConcept
        self.run_query("CREATE CONSTRAINT related_concept_uri_constraint IF NOT EXISTS FOR (r:RelatedConcept) REQUIRE r.uri IS UNIQUE")
        
        logger.info("Đã tạo các ràng buộc và chỉ mục cho Neo4j")
    
    def clear_database(self):
        """
        Xóa tất cả dữ liệu trong database
        """
        self.run_query("MATCH (n) DETACH DELETE n")
        logger.info("Đã xóa tất cả dữ liệu trong database")

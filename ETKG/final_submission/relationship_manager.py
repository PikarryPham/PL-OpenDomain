import logging
import sys
sys.path.append('/home/ubuntu')
from config import *
from connector import Neo4jConnector
from utils.data_loader import get_current_timestamp

logger = logging.getLogger(__name__)

class RelationshipManager:
    """
    Lớp quản lý relationship trong Neo4j
    """
    
    def __init__(self, connector):
        """
        Khởi tạo RelationshipManager
        
        Args:
            connector (Neo4jConnector): Đối tượng kết nối với Neo4j
        """
        self.connector = connector
    
    def create_or_update_user_entry_relationship(self, user_id, entry_id, timestamp):
        """
        Tạo hoặc cập nhật relationship HAS giữa User và Entry
        
        Args:
            user_id (str): ID của User
            entry_id (str): ID của Entry
            timestamp (str): Thời gian của relationship
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            query = """
            MATCH (u:User {user_id: $user_id})
            MATCH (e:Entry {entry_id: $entry_id})
            MERGE (u)-[r:HAS]->(e)
            ON CREATE SET r.timestamp = $timestamp
            ON MATCH SET r.timestamp = $timestamp
            RETURN r
            """
            parameters = {
                'user_id': user_id,
                'entry_id': entry_id,
                'timestamp': timestamp
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật relationship HAS giữa User {user_id} và Entry {entry_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật relationship HAS giữa User và Entry: {str(e)}")
            return False
    
    def create_or_update_user_option_relationship(self, user_id, option_id, timestamp, order):
        """
        Tạo hoặc cập nhật relationship CHOOSE giữa User và Option
        
        Args:
            user_id (str): ID của User
            option_id (str): ID của Option
            timestamp (str): Thời gian của relationship
            order (int): Thứ tự lựa chọn
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            query = """
            MATCH (u:User {user_id: $user_id})
            MATCH (o:Option {id: $option_id})
            MERGE (u)-[r:CHOOSE]->(o)
            ON CREATE SET r.timestamp = $timestamp, r.order = $order
            ON MATCH SET r.timestamp = $timestamp, r.order = $order
            RETURN r
            """
            parameters = {
                'user_id': user_id,
                'option_id': option_id,
                'timestamp': timestamp,
                'order': order
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật relationship CHOOSE giữa User {user_id} và Option {option_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật relationship CHOOSE giữa User và Option: {str(e)}")
            return False
    
    def create_or_update_question_option_relationship(self, question_id, option_id, timestamp):
        """
        Tạo hoặc cập nhật relationship HAS giữa Question và Option
        
        Args:
            question_id (str): ID của Question
            option_id (str): ID của Option
            timestamp (str): Thời gian của relationship
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            query = """
            MATCH (q:Question {id: $question_id})
            MATCH (o:Option {id: $option_id})
            MERGE (q)-[r:HAS]->(o)
            ON CREATE SET r.timestamp = $timestamp
            ON MATCH SET r.timestamp = $timestamp
            RETURN r
            """
            parameters = {
                'question_id': question_id,
                'option_id': option_id,
                'timestamp': timestamp
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật relationship HAS giữa Question {question_id} và Option {option_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật relationship HAS giữa Question và Option: {str(e)}")
            return False
    
    def create_or_update_entry_topic_relationship(self, entry_id, topic_uri, weight_duration, weight_view):
        """
        Tạo hoặc cập nhật relationship RELATE_TO giữa Entry và Topic
        
        Args:
            entry_id (str): ID của Entry
            topic_uri (str): URI của Topic
            weight_duration (int): Thời gian xem trang
            weight_view (int): Số lần xem trang
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            timestamp = get_current_timestamp()
            query = """
            MATCH (e:Entry {entry_id: $entry_id})
            MATCH (t:Topic {uri: $topic_uri})
            MERGE (e)-[r:RELATE_TO]->(t)
            ON CREATE SET r.weightDurationPerSession = $weight_duration,
                          r.weightView = $weight_view,
                          r.timestamp = $timestamp
            ON MATCH SET r.weightDurationPerSession = $weight_duration,
                         r.weightView = $weight_view,
                         r.timestamp = $timestamp
            RETURN r
            """
            parameters = {
                'entry_id': entry_id,
                'topic_uri': topic_uri,
                'weight_duration': weight_duration,
                'weight_view': weight_view,
                'timestamp': timestamp
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật relationship RELATE_TO giữa Entry {entry_id} và Topic {topic_uri}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật relationship RELATE_TO giữa Entry và Topic: {str(e)}")
            return False
    
    def create_or_update_category_topic_relationship(self, category_uri, topic_uri):
        """
        Tạo hoặc cập nhật relationship HAS_ROOT giữa Category và Topic
        
        Args:
            category_uri (str): URI của Category
            topic_uri (str): URI của Topic
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            timestamp = get_current_timestamp()
            query = """
            MATCH (c:Category {uri: $category_uri})
            MATCH (t:Topic {uri: $topic_uri})
            MERGE (c)-[r:HAS_ROOT]->(t)
            ON CREATE SET r.timestamp = $timestamp
            ON MATCH SET r.timestamp = $timestamp
            RETURN r
            """
            parameters = {
                'category_uri': category_uri,
                'topic_uri': topic_uri,
                'timestamp': timestamp
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật relationship HAS_ROOT giữa Category {category_uri} và Topic {topic_uri}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật relationship HAS_ROOT giữa Category và Topic: {str(e)}")
            return False
    
    def create_or_update_related_concept_category_relationship(self, related_concept_uri, category_uri):
        """
        Tạo hoặc cập nhật relationship HAS_PARENT giữa RelatedConcept và Category
        
        Args:
            related_concept_uri (str): URI của RelatedConcept
            category_uri (str): URI của Category
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            timestamp = get_current_timestamp()
            query = """
            MATCH (r:RelatedConcept {uri: $related_concept_uri})
            MATCH (c:Category {uri: $category_uri})
            MERGE (r)-[rel:HAS_PARENT]->(c)
            ON CREATE SET rel.timestamp = $timestamp
            ON MATCH SET rel.timestamp = $timestamp
            RETURN rel
            """
            parameters = {
                'related_concept_uri': related_concept_uri,
                'category_uri': category_uri,
                'timestamp': timestamp
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật relationship HAS_PARENT giữa RelatedConcept {related_concept_uri} và Category {category_uri}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật relationship HAS_PARENT giữa RelatedConcept và Category: {str(e)}")
            return False

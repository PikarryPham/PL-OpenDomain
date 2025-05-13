import logging
import sys
sys.path.append('/home/ubuntu')
from config import *
from connector import Neo4jConnector
from utils.data_loader import get_current_timestamp

logger = logging.getLogger(__name__)

class NodeManager:
    """
    Lớp quản lý node trong Neo4j
    """
    
    def __init__(self, connector):
        """
        Khởi tạo NodeManager
        
        Args:
            connector (Neo4jConnector): Đối tượng kết nối với Neo4j
        """
        self.connector = connector
    
    def create_or_update_user_node(self, user):
        """
        Tạo hoặc cập nhật User Node
        
        Args:
            user (User): Đối tượng User
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            properties = user.get_cypher_properties()
            query = """
            MERGE (u:User {user_id: $user_id})
            ON CREATE SET u += $properties
            ON MATCH SET u += $properties
            RETURN u
            """
            parameters = {
                'user_id': user.user_id,
                'properties': properties
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật User Node với ID: {user.user_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật User Node: {str(e)}")
            return False
    
    def create_or_update_question_node(self, question):
        """
        Tạo hoặc cập nhật Question Node
        
        Args:
            question (Question): Đối tượng Question
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            properties = question.get_cypher_properties()
            query = """
            MERGE (q:Question {id: $id})
            ON CREATE SET q += $properties
            ON MATCH SET q += $properties
            RETURN q
            """
            parameters = {
                'id': question.id,
                'properties': properties
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật Question Node với ID: {question.id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật Question Node: {str(e)}")
            return False
    
    def create_or_update_option_node(self, option):
        """
        Tạo hoặc cập nhật Option Node
        
        Args:
            option (Option): Đối tượng Option
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            properties = option.get_cypher_properties()
            query = """
            MERGE (o:Option {id: $id})
            ON CREATE SET o += $properties
            ON MATCH SET o += $properties
            RETURN o
            """
            parameters = {
                'id': option.id,
                'properties': properties
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật Option Node với ID: {option.id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật Option Node: {str(e)}")
            return False
    
    def create_or_update_entry_node(self, entry):
        """
        Tạo hoặc cập nhật Entry Node
        
        Args:
            entry (Entry): Đối tượng Entry
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            properties = entry.get_cypher_properties()
            query = """
            MERGE (e:Entry {entry_id: $entry_id})
            ON CREATE SET e += $properties
            ON MATCH SET e += $properties
            RETURN e
            """
            parameters = {
                'entry_id': entry.entry_id,
                'properties': properties
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật Entry Node với ID: {entry.entry_id}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật Entry Node: {str(e)}")
            return False
    
    def create_or_update_concept_node(self, concept):
        """
        Tạo hoặc cập nhật Concept Node (Topic, Category, RelatedConcept)
        
        Args:
            concept (Concept): Đối tượng Concept
            
        Returns:
            bool: True nếu thành công, False nếu thất bại
        """
        try:
            properties = concept.get_cypher_properties()
            concept_type = concept.concept_type
            
            query = f"""
            MERGE (c:{concept_type} {{uri: $uri}})
            ON CREATE SET c += $properties
            ON MATCH SET c += $properties
            RETURN c
            """
            parameters = {
                'uri': concept.uri,
                'properties': properties
            }
            result = self.connector.run_query(query, parameters)
            logger.info(f"Đã tạo hoặc cập nhật {concept_type} Node với URI: {concept.uri}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi tạo hoặc cập nhật {concept.concept_type} Node: {str(e)}")
            return False

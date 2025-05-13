import logging
import sys
import json
from datetime import datetime
sys.path.append('/home/ubuntu')

from config import *
from db_connector import Neo4jConnector
from node_manager import NodeManager
from relationship_manager import RelationshipManager
from utils.data_loader import (
    load_users_data, load_questions_data, load_options_data,
    load_history_learning_data, load_final_sample_output,
    batch_data, get_current_timestamp
)
from utils.data_processor import DataProcessor
from models.user import User
from models.question import Question
from models.option import Option
from models.entry import Entry
from models.concept import Concept

# Cấu hình logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=LOG_FILE,
    filemode='a'
)
logger = logging.getLogger(__name__)

class TKGBuilder:
    """
    Lớp xây dựng Temporal Knowledge Graph
    """
    
    def __init__(self):
        """
        Khởi tạo TKGBuilder
        """
        self.connector = Neo4jConnector()
        self.node_manager = NodeManager(self.connector)
        self.relationship_manager = RelationshipManager(self.connector)
        self.data_processor = DataProcessor()
    
    def setup_database(self):
        """
        Thiết lập database
        """
        logger.info("Bắt đầu thiết lập database")
        self.connector.create_constraints_and_indexes()
        logger.info("Đã thiết lập database thành công")
    
    def process_users(self, users_data):
        """
        Xử lý dữ liệu người dùng
        
        Args:
            users_data (list): Danh sách dữ liệu người dùng
        """
        logger.info(f"Bắt đầu xử lý {len(users_data)} người dùng")
        
        for user_data in users_data:
            # Xử lý dữ liệu người dùng
            processed_user_data = self.data_processor.process_user_data(user_data)
            user = User(processed_user_data)
            
            # Tạo hoặc cập nhật User Node
            self.node_manager.create_or_update_user_node(user)
        
        logger.info("Đã xử lý xong dữ liệu người dùng")
    
    def process_questions_and_options(self, questions_data, options_data):
        """
        Xử lý dữ liệu câu hỏi và lựa chọn
        
        Args:
            questions_data (list): Danh sách dữ liệu câu hỏi
            options_data (list): Danh sách dữ liệu lựa chọn
        """
        logger.info(f"Bắt đầu xử lý {len(questions_data)} câu hỏi và {len(options_data)} lựa chọn")
        
        # Xử lý câu hỏi
        for question_data in questions_data:
            processed_question_data = self.data_processor.process_question_data(question_data)
            question = Question(processed_question_data)
            
            # Tạo hoặc cập nhật Question Node
            self.node_manager.create_or_update_question_node(question)
        
        # Xử lý lựa chọn và mối quan hệ với câu hỏi
        for option_data in options_data:
            processed_option_data = self.data_processor.process_option_data(option_data)
            option = Option(processed_option_data)
            
            # Tạo hoặc cập nhật Option Node
            self.node_manager.create_or_update_option_node(option)
            
            # Tạo hoặc cập nhật relationship giữa Question và Option
            if option.question_id:
                timestamp = get_current_timestamp()
                self.relationship_manager.create_or_update_question_option_relationship(
                    option.question_id, option.id, timestamp
                )
        
        logger.info("Đã xử lý xong dữ liệu câu hỏi và lựa chọn")
    
    def process_user_options(self, users_data, questions_data, options_data):
        """
        Xử lý mối quan hệ giữa người dùng và lựa chọn
        
        Args:
            users_data (list): Danh sách dữ liệu người dùng
            questions_data (list): Danh sách dữ liệu câu hỏi
            options_data (list): Danh sách dữ liệu lựa chọn
        """
        logger.info("Bắt đầu xử lý mối quan hệ giữa người dùng và lựa chọn")
        
        for user_data in users_data:
            user_id = user_data.get('user_id')
            updated_time = user_data.get('updated_time')
            
            # Xử lý preferred_areas
            for area in user_data.get('preferred_areas', []):
                option_text = area.get('option')
                order = area.get('order')
                
                # Tìm câu hỏi có form là preferred_areas
                question = self.data_processor.find_question_by_form('preferred_areas', questions_data)
                if question:
                    # Tìm lựa chọn có option_text tương ứng
                    options = self.data_processor.find_options_for_question(question.get('id'), options_data)
                    for option in options:
                        if option.get('option_text') == option_text:
                            # Tạo hoặc cập nhật relationship giữa User và Option
                            self.relationship_manager.create_or_update_user_option_relationship(
                                user_id, option.get('id'), updated_time, order
                            )
            
            # Xử lý preferred_content_types
            for content_type in user_data.get('preferred_content_types', []):
                option_text = content_type.get('option')
                order = content_type.get('order')
                
                # Tìm câu hỏi có form là preferred_content_types
                question = self.data_processor.find_question_by_form('preferred_content_types', questions_data)
                if question:
                    # Tìm lựa chọn có option_text tương ứng
                    options = self.data_processor.find_options_for_question(question.get('id'), options_data)
                    for option in options:
                        if option.get('option_text') == option_text:
                            # Tạo hoặc cập nhật relationship giữa User và Option
                            self.relationship_manager.create_or_update_user_option_relationship(
                                user_id, option.get('id'), updated_time, order
                            )
            
            # Xử lý preferred_learn_style
            for learn_style in user_data.get('preferred_learn_style', []):
                option_text = learn_style.get('option')
                order = learn_style.get('order')
                
                # Tìm câu hỏi có form là preferred_learn_style
                question = self.data_processor.find_question_by_form('preferred_learn_style', questions_data)
                if question:
                    # Tìm lựa chọn có option_text tương ứng
                    options = self.data_processor.find_options_for_question(question.get('id'), options_data)
                    for option in options:
                        if option.get('option_text') == option_text:
                            # Tạo hoặc cập nhật relationship giữa User và Option
                            self.relationship_manager.create_or_update_user_option_relationship(
                                user_id, option.get('id'), updated_time, order
                            )
            
            # Xử lý education_lv
            for education in user_data.get('education_lv', []):
                option_text = education.get('option')
                order = education.get('order')
                
                # Tìm câu hỏi có form là education_lv
                question = self.data_processor.find_question_by_form('education_lv', questions_data)
                if question:
                    # Tìm lựa chọn có option_text tương ứng
                    options = self.data_processor.find_options_for_question(question.get('id'), options_data)
                    for option in options:
                        if option.get('option_text') == option_text:
                            # Tạo hoặc cập nhật relationship giữa User và Option
                            self.relationship_manager.create_or_update_user_option_relationship(
                                user_id, option.get('id'), updated_time, order
                            )
        
        logger.info("Đã xử lý xong mối quan hệ giữa người dùng và lựa chọn")
    
    def process_entries(self, entries_data):
        """
        Xử lý dữ liệu entry
        
        Args:
            entries_data (list): Danh sách dữ liệu entry
        """
        logger.info(f"Bắt đầu xử lý {len(entries_data)} entry")
        
        for entry_data in entries_data:
            # Xử lý dữ liệu entry
            processed_entry_data = self.data_processor.process_entry_data(entry_data)
            entry = Entry(processed_entry_data)
            
            # Tạo hoặc cập nhật Entry Node
            self.node_manager.create_or_update_entry_node(entry)
            
            # Tạo hoặc cập nhật relationship giữa User và Entry
            if entry.user_id:
                self.relationship_manager.create_or_update_user_entry_relationship(
                    entry.user_id, entry.entry_id, entry.timestamp
                )
        
        logger.info("Đã xử lý xong dữ liệu entry")
    
    def process_concepts(self, concepts_data):
        """
        Xử lý dữ liệu concept
        
        Args:
            concepts_data (dict): Dữ liệu concept
        """
        logger.info("Bắt đầu xử lý dữ liệu concept")
        
        result = concepts_data.get('result', [])
        for item in result:
            entry_id = item.get('entry_id')
            pages = item.get('pages', [])
            
            for page in pages:
                # Xử lý Topic (Level 1)
                topic_data = page.get('topic', {})
                if topic_data and topic_data.get('uri'):
                    topic = Concept.create_topic(topic_data)
                    self.node_manager.create_or_update_concept_node(topic)
                    
                    # Tạo hoặc cập nhật relationship giữa Entry và Topic
                    entry_data = next((e for e in load_history_learning_data() if e.get('entry_id') == entry_id), None)
                    if entry_data:
                        weight_duration = entry_data.get('capped_time_on_page', 0)
                        weight_view = entry_data.get('pageview_count', 0)
                        self.relationship_manager.create_or_update_entry_topic_relationship(
                            entry_id, topic.uri, weight_duration, weight_view
                        )
                
                # Xử lý Category (Level 2)
                category_data = page.get('category', {})
                if category_data and category_data.get('uri'):
                    category = Concept.create_category(category_data)
                    self.node_manager.create_or_update_concept_node(category)
                    
                    # Tạo hoặc cập nhật relationship giữa Category và Topic
                    if topic_data and topic_data.get('uri'):
                        self.relationship_manager.create_or_update_category_topic_relationship(
                            category.uri, topic.uri
                        )
                
                # Xử lý RelatedConcept (Level 3)
                related_concept_data = {
                    'uri': page.get('relatedConcept', {}).get('value'),
                    'label': page.get('relatedConceptLabel', {}).get('value'),
                    'language': page.get('relatedConceptLabel', {}).get('xml:lang'),
                    'relationshipType': page.get('relationshipType', {}).get('value'),
                    'abstract': page.get('abstract'),
                    'comment': page.get('comment')
                }
                
                if related_concept_data.get('uri'):
                    related_concept = Concept.create_related_concept(related_concept_data)
                    self.node_manager.create_or_update_concept_node(related_concept)
                    
                    # Tạo hoặc cập nhật relationship giữa RelatedConcept và Category
                    if category_data and category_data.get('uri'):
                        self.relationship_manager.create_or_update_related_concept_category_relationship(
                            related_concept.uri, category.uri
                        )
        
        logger.info("Đã xử lý xong dữ liệu concept")
    
    def build_knowledge_graph(self):
        """
        Xây dựng Knowledge Graph
        """
        logger.info("Bắt đầu xây dựng Knowledge Graph")
        
        # Thiết lập database
        self.setup_database()
        
        # Đọc dữ liệu
        users_data = load_users_data()
        questions_data = load_questions_data()
        options_data = load_options_data()
        entries_data = load_history_learning_data()
        concepts_data = load_final_sample_output()
        
        # Xử lý dữ liệu
        self.process_users(users_data)
        self.process_questions_and_options(questions_data, options_data)
        self.process_user_options(users_data, questions_data, options_data)
        self.process_entries(entries_data)
        self.process_concepts(concepts_data)
        
        logger.info("Đã xây dựng xong Knowledge Graph")
    
    def close(self):
        """
        Đóng kết nối với Neo4j
        """
        self.connector.close()
        logger.info("Đã đóng kết nối với Neo4j")

def main():
    """
    Hàm chính
    """
    try:
        # Khởi tạo TKGBuilder
        tkg_builder = TKGBuilder()
        
        # Xây dựng Knowledge Graph
        tkg_builder.build_knowledge_graph()
        
        # Đóng kết nối
        tkg_builder.close()
        
        print("Đã xây dựng xong Knowledge Graph")
    except Exception as e:
        logger.error(f"Lỗi khi xây dựng Knowledge Graph: {str(e)}")
        print(f"Lỗi khi xây dựng Knowledge Graph: {str(e)}")

if __name__ == "__main__":
    main()

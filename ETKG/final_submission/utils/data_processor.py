import logging
import sys
sys.path.append('/home/ubuntu')
from config import *

logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Lớp xử lý và chuẩn hóa dữ liệu từ các file JSON
    """
    
    @staticmethod
    def process_user_data(user):
        """
        Xử lý dữ liệu người dùng
        
        Args:
            user (dict): Dữ liệu người dùng
            
        Returns:
            dict: Dữ liệu người dùng đã xử lý
        """
        processed_user = {
            'user_id': user.get('user_id'),
            'username': user.get('username'),
            'email': user.get('email'),
            'created_time_original': user.get('created_time_original'),
            'updated_time': user.get('updated_time'),
            'preferred_areas': user.get('preferred_areas', []),
            'preferred_content_types': user.get('preferred_content_types', []),
            'preferred_learn_style': user.get('preferred_learn_style', []),
            'education_lv': user.get('education_lv', [])
        }
        return processed_user
    
    @staticmethod
    def process_question_data(question):
        """
        Xử lý dữ liệu câu hỏi
        
        Args:
            question (dict): Dữ liệu câu hỏi
            
        Returns:
            dict: Dữ liệu câu hỏi đã xử lý
        """
        processed_question = {
            'id': question.get('id'),
            'question_text': question.get('question_text'),
            'type': question.get('type'),
            'form': question.get('form'),
            'created_at': question.get('created_at')
        }
        return processed_question
    
    @staticmethod
    def process_option_data(option):
        """
        Xử lý dữ liệu lựa chọn
        
        Args:
            option (dict): Dữ liệu lựa chọn
            
        Returns:
            dict: Dữ liệu lựa chọn đã xử lý
        """
        processed_option = {
            'id': option.get('id'),
            'question_id': option.get('question_id'),
            'option_text': option.get('option_text')
        }
        return processed_option
    
    @staticmethod
    def process_entry_data(entry):
        """
        Xử lý dữ liệu entry
        
        Args:
            entry (dict): Dữ liệu entry
            
        Returns:
            dict: Dữ liệu entry đã xử lý
        """
        processed_entry = {
            'entry_id': entry.get('entry_id'),
            'url': entry.get('url'),
            'title': entry.get('title'),
            'timestamp': entry.get('timestamp'),
            'pageview_count': entry.get('pageview_count'),
            'window_time_details_meaning': entry.get('window_time_details_meaning'),
            'window_time_overall_meaning': entry.get('window_time_overall_meaning'),
            'capped_time_on_page': entry.get('capped_time_on_page'),
            'session_id': entry.get('session_id'),
            'user_id': entry.get('user_id'),
            'visible_content': entry.get('visible_content'),
            'tmp_keywords': entry.get('tmp_keywords', []),
            'exact_keywords': entry.get('exact_keywords', [])
        }
        return processed_entry
    
    @staticmethod
    def process_concept_data(concept_data):
        """
        Xử lý dữ liệu concept
        
        Args:
            concept_data (dict): Dữ liệu concept
            
        Returns:
            tuple: (topic, category, related_concept) đã xử lý
        """
        # Xử lý Topic (Level 1)
        topic = {
            'uri': concept_data.get('topic', {}).get('uri'),
            'label': concept_data.get('topic', {}).get('label')
        }
        
        # Xử lý Category (Level 2)
        category = {
            'uri': concept_data.get('category', {}).get('uri'),
            'label': concept_data.get('category', {}).get('label')
        }
        
        # Xử lý RelatedConcept (Level 3)
        related_concept = {
            'uri': concept_data.get('relatedConcept', {}).get('value'),
            'label': concept_data.get('relatedConceptLabel', {}).get('value'),
            'language': concept_data.get('relatedConceptLabel', {}).get('xml:lang'),
            'relationshipType': concept_data.get('relationshipType', {}).get('value'),
            'abstract': concept_data.get('abstract'),
            'comment': concept_data.get('comment')
        }
        
        return topic, category, related_concept
    
    @staticmethod
    def find_options_for_question(question_id, options_data):
        """
        Tìm các lựa chọn cho một câu hỏi
        
        Args:
            question_id (str): ID của câu hỏi
            options_data (list): Danh sách các lựa chọn
            
        Returns:
            list: Danh sách các lựa chọn cho câu hỏi
        """
        return [option for option in options_data if option.get('question_id') == question_id]
    
    @staticmethod
    def find_question_by_form(form, questions_data):
        """
        Tìm câu hỏi theo form
        
        Args:
            form (str): Loại form
            questions_data (list): Danh sách các câu hỏi
            
        Returns:
            dict: Câu hỏi tương ứng với form
        """
        for question in questions_data:
            if question.get('form') == form:
                return question
        return None
    
    @staticmethod
    def find_option_by_text(option_text, options_data):
        """
        Tìm lựa chọn theo nội dung
        
        Args:
            option_text (str): Nội dung lựa chọn
            options_data (list): Danh sách các lựa chọn
            
        Returns:
            dict: Lựa chọn tương ứng với nội dung
        """
        for option in options_data:
            if option.get('option_text') == option_text:
                return option
        return None
    
    @staticmethod
    def find_entries_for_user(user_id, entries_data):
        """
        Tìm các entry cho một người dùng
        
        Args:
            user_id (str): ID của người dùng
            entries_data (list): Danh sách các entry
            
        Returns:
            list: Danh sách các entry cho người dùng
        """
        return [entry for entry in entries_data if entry.get('user_id') == user_id]
    
    @staticmethod
    def find_concepts_for_entry(entry_id, concepts_data):
        """
        Tìm các concept cho một entry
        
        Args:
            entry_id (str): ID của entry
            concepts_data (dict): Dữ liệu concept
            
        Returns:
            list: Danh sách các concept cho entry
        """
        result = concepts_data.get('result', [])
        for item in result:
            if item.get('entry_id') == entry_id:
                return item.get('pages', [])
        return []

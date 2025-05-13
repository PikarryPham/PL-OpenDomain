import sys
sys.path.append('/home/ubuntu')
from config import *

class Entry:
    """
    Model cho Entry Node
    """
    def __init__(self, entry_data):
        self.entry_id = entry_data.get('entry_id')
        self.url = entry_data.get('url')
        self.title = entry_data.get('title')
        self.timestamp = entry_data.get('timestamp')
        self.pageview_count = entry_data.get('pageview_count')
        self.window_time_details_meaning = entry_data.get('window_time_details_meaning')
        self.window_time_overall_meaning = entry_data.get('window_time_overall_meaning')
        self.capped_time_on_page = entry_data.get('capped_time_on_page')
        self.session_id = entry_data.get('session_id')
        self.user_id = entry_data.get('user_id')
        self.visible_content = entry_data.get('visible_content')
        self.exact_keywords = entry_data.get('exact_keywords', [])
        self.tmp_keywords = entry_data.get('tmp_keywords', [])
        self.exact_keywords = entry_data.get('exact_keywords', [])
    def to_dict(self):
        """
        Chuyển đổi đối tượng thành dictionary
        
        Returns:
            dict: Dictionary chứa thông tin của Entry
        """
        return {
            'entry_id': self.entry_id,
            'url': self.url,
            'title': self.title,
            'timestamp': self.timestamp,
            'pageview_count': self.pageview_count,
            'window_time_details_meaning': self.window_time_details_meaning,
            'window_time_overall_meaning': self.window_time_overall_meaning,
            'capped_time_on_page': self.capped_time_on_page,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'tmp_keywords': self.tmp_keywords,
            'exact_keywords': self.exact_keywords
        }
    
    def get_cypher_properties(self):
        """
        Lấy các thuộc tính cho Cypher query
        
        Returns:
            dict: Dictionary chứa các thuộc tính cho Cypher query
        """
        properties = self.to_dict()
        # Loại bỏ các giá trị None
        return {k: v for k, v in properties.items() if v is not None}

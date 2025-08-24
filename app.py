from flask import Flask, request, jsonify, send_from_directory
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# YouTube API configuration - REPLACE WITH YOUR ACTUAL API KEY
YOUTUBE_API_KEY = 'AIzaSyCzZkx9s5I_4EPXESWcYxLEtnn7R7WRgxM'
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

class YouTubeAnalyzer:
    def __init__(self):
        self.youtube = youtube
    
    def search_youtuber(self, query, max_results=10):
        """Search for a YouTuber and get their channel info"""
        try:
            # Search for channels
            search_response = self.youtube.search().list(
                q=query,
                type='channel',
                part='id,snippet',
                maxResults=5
            ).execute()
            
            if not search_response['items']:
                return {'error': 'No channels found'}
            
            # Get the first channel (most relevant)
            channel = search_response['items'][0]
            channel_id = channel['id']['channelId']
            channel_title = channel['snippet']['title']
            
            # Get channel statistics
            channel_stats = self.youtube.channels().list(
                part='statistics,snippet',
                id=channel_id
            ).execute()
            
            channel_info = channel_stats['items'][0]
            
            # Get latest videos from this channel
            videos_data = self.get_channel_videos(channel_id, max_results)
            
            return {
                'channel_info': {
                    'id': channel_id,
                    'title': channel_title,
                    'subscriber_count': int(channel_info['statistics'].get('subscriberCount', 0)),
                    'video_count': int(channel_info['statistics'].get('videoCount', 0)),
                    'view_count': int(channel_info['statistics'].get('viewCount', 0)),
                    'thumbnail': channel_info['snippet']['thumbnails']['high']['url']
                },
                'videos': videos_data
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_channel_videos(self, channel_id, max_results=10):
        """Get latest videos from a channel with detailed statistics"""
        try:
            # Get video IDs from channel
            search_response = self.youtube.search().list(
                channelId=channel_id,
                type='video',
                order='date',
                part='id,snippet',
                maxResults=max_results
            ).execute()
            
            video_ids = [item['id']['videoId'] for item in search_response['items']]
            
            if not video_ids:
                return []
            
            # Get detailed statistics for videos
            videos_response = self.youtube.videos().list(
                part='statistics,snippet,contentDetails',
                id=','.join(video_ids)
            ).execute()
            
            videos_data = []
            for video in videos_response['items']:
                video_data = self.analyze_video(video)
                videos_data.append(video_data)
            
            # Sort by engagement rate (descending)
            videos_data.sort(key=lambda x: x['engagement_rate'], reverse=True)
            
            return videos_data
            
        except Exception as e:
            print(f"Error getting channel videos: {e}")
            return []
    
    def analyze_video(self, video):
        """Analyze individual video and calculate metrics"""
        stats = video['statistics']
        snippet = video['snippet']
        
        # Extract basic stats
        views = int(stats.get('viewCount', 0))
        likes = int(stats.get('likeCount', 0))
        comments = int(stats.get('commentCount', 0))
        
        # Calculate engagement rate
        total_engagement = likes + comments
        engagement_rate = (total_engagement / views * 100) if views > 0 else 0
        
        # Calculate days since publish
        publish_date = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
        days_since_publish = (datetime.now(publish_date.tzinfo) - publish_date).days
        
        # Get video duration
        duration = self.parse_duration(video['contentDetails']['duration'])
        
        # Analyze video performance
        analysis = self.rule_based_analysis(views, engagement_rate, days_since_publish, snippet)
        
        return {
            'id': video['id'],
            'title': snippet['title'],
            'description': snippet['description'][:200] + '...' if len(snippet['description']) > 200 else snippet['description'],
            'thumbnail': snippet['thumbnails']['maxres']['url'] if 'maxres' in snippet['thumbnails'] else snippet['thumbnails']['high']['url'],
            'published_at': snippet['publishedAt'],
            'views': views,
            'likes': likes,
            'comments': comments,
            'engagement_rate': round(engagement_rate, 2),
            'days_since_publish': days_since_publish,
            'duration': duration,
            'url': f"https://youtube.com/watch?v={video['id']}",
            'analysis': analysis
        }
    
    def parse_duration(self, duration):
        """Convert ISO 8601 duration to readable format"""
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration)
        if match:
            hours, minutes, seconds = match.groups()
            hours = int(hours) if hours else 0
            minutes = int(minutes) if minutes else 0
            seconds = int(seconds) if seconds else 0
            
            if hours > 0:
                return f"{hours}:{minutes:02d}:{seconds:02d}"
            else:
                return f"{minutes}:{seconds:02d}"
        return "0:00"
    
    def rule_based_analysis(self, views, engagement_rate, days_since_publish, snippet):
        """Rule-based analysis similar to your n8n workflow"""
        # Calculate viral score (1-10)
        viral_score = 1
        
        # Views factor
        if views > 1000000:
            viral_score += 4
        elif views > 100000:
            viral_score += 3
        elif views > 10000:
            viral_score += 2
        elif views > 1000:
            viral_score += 1
        
        # Engagement rate factor
        if engagement_rate > 8:
            viral_score += 3
        elif engagement_rate > 5:
            viral_score += 2
        elif engagement_rate > 2:
            viral_score += 1
        
        # Recency factor (newer videos get bonus)
        if days_since_publish <= 7:
            viral_score += 2
        elif days_since_publish <= 30:
            viral_score += 1
        
        viral_score = min(viral_score, 10)
        
        # Content category detection
        title_lower = snippet['title'].lower()
        description_lower = snippet['description'].lower()
        
        categories = {
            'Gaming': ['game', 'gaming', 'play', 'gameplay', 'stream'],
            'Tech': ['tech', 'review', 'unbox', 'smartphone', 'laptop', 'ai'],
            'Entertainment': ['funny', 'comedy', 'react', 'challenge', 'prank'],
            'Education': ['tutorial', 'learn', 'how to', 'explain', 'guide'],
            'Lifestyle': ['vlog', 'daily', 'routine', 'life', 'travel'],
            'Music': ['music', 'song', 'cover', 'sing', 'dance'],
            'News': ['news', 'update', 'breaking', 'report', 'current']
        }
        
        content_category = 'General'
        for category, keywords in categories.items():
            if any(keyword in title_lower or keyword in description_lower for keyword in keywords):
                content_category = category
                break
        
        # Brand safety score (1-10, higher is safer)
        risky_keywords = ['controversy', 'drama', 'exposed', 'scandal', 'beef', 'diss']
        brand_safety_score = 10
        
        for keyword in risky_keywords:
            if keyword in title_lower or keyword in description_lower:
                brand_safety_score -= 2
        
        brand_safety_score = max(brand_safety_score, 1)
        
        # Trend indicators
        trend_indicators = []
        if engagement_rate > 5:
            trend_indicators.append('high-engagement')
        if views > 100000:
            trend_indicators.append('high-views')
        if days_since_publish <= 7:
            trend_indicators.append('recent-content')
        if viral_score >= 8:
            trend_indicators.append('viral-potential')
        
        # Engagement quality
        if engagement_rate > 8:
            engagement_quality = 'excellent'
        elif engagement_rate > 5:
            engagement_quality = 'high'
        elif engagement_rate > 2:
            engagement_quality = 'medium'
        else:
            engagement_quality = 'low'
        
        # Recommended action
        if viral_score >= 8 and brand_safety_score >= 8:
            recommended_action = 'High priority for partnership'
        elif viral_score >= 6 and brand_safety_score >= 7:
            recommended_action = 'Monitor for partnership opportunities'
        elif viral_score >= 4:
            recommended_action = 'Consider for content collaboration'
        else:
            recommended_action = 'Not recommended for partnership'
        
        return {
            'viral_score': viral_score,
            'content_category': content_category,
            'brand_safety_score': brand_safety_score,
            'trend_indicators': trend_indicators,
            'engagement_quality': engagement_quality,
            'recommended_action': recommended_action
        }

# Initialize analyzer
analyzer = YouTubeAnalyzer()

@app.route('/')
def index():
    """Serve the HTML dashboard"""
    return send_from_directory('.', 'index.html')

@app.route('/api/search', methods=['POST'])
def search_youtuber():
    """API endpoint to search for YouTuber and analyze their videos"""
    data = request.json
    query = data.get('query', '').strip()
    max_results = data.get('max_results', 10)
    
    if not query:
        return jsonify({'error': 'Please provide a search query'}), 400
    
    # Analyze YouTuber
    result = analyzer.search_youtuber(query, max_results)
    
    if 'error' in result:
        return jsonify(result), 400
    
    return jsonify(result)

@app.route('/api/video/<video_id>')
def get_video_details(video_id):
    """Get detailed analysis for a specific video"""
    try:
        video_response = youtube.videos().list(
            part='statistics,snippet,contentDetails',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            return jsonify({'error': 'Video not found'}), 404
        
        video_data = analyzer.analyze_video(video_response['items'][0])
        return jsonify(video_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Check API key
    if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY':
        print("🚨 WARNING: Please set your YouTube API key!")
        print("1. Go to: https://console.developers.google.com/")
        print("2. Enable YouTube Data API v3")
        print("3. Create credentials -> API Key")
        print("4. Replace 'YOUR_YOUTUBE_API_KEY' in app.py with your actual key")
        print()
    
    print("🎥 YouTube Analytics Dashboard")
    print("📡 Server starting on: http://localhost:5000")
    print("📊 Place your index.html file in the same directory as app.py")
    
    app.run(debug=True, port=5000)
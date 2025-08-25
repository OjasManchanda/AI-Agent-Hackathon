from flask import Flask, request, jsonify, send_from_directory, render_template_string
from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re
import os
import json
import threading
import time
from collections import Counter, defaultdict
from flask_cors import CORS

# AI Libraries
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    from textblob import TextBlob
    import spacy
    AI_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  AI libraries not installed: {e}")
    print("Install with: pip install vaderSentiment textblob spacy")
    AI_FEATURES_AVAILABLE = False

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    SCHEDULER_AVAILABLE = True
except ImportError:
    print("⚠️  Scheduler not available. Install with: pip install apscheduler")
    SCHEDULER_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# YouTube API configuration - FIXED API KEY
YOUTUBE_API_KEY = 'AIzaSyAZGZp3p2VRWcczLzgqF6FigM93iaf9CyU'  # Your working API key
youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

# Initialize AI models if available
if AI_FEATURES_AVAILABLE:
    analyzer = SentimentIntensityAnalyzer()
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Warning: spaCy model not found. Run: python -m spacy download en_core_web_sm")
        nlp = None
else:
    analyzer = None
    nlp = None

class AIAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = analyzer
        self.nlp = nlp
        
    def analyze_content_themes(self, texts):
        """Extract main themes and topics from content using NLP"""
        if not self.nlp or not AI_FEATURES_AVAILABLE:
            return self._fallback_theme_analysis(texts)
            
        all_entities = []
        all_keywords = []
        
        for text in texts:
            if len(text) > 1000:  # Limit text length for processing
                text = text[:1000]
                
            doc = self.nlp(text)
            
            # Extract named entities
            entities = [(ent.text.lower(), ent.label_) for ent in doc.ents 
                       if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'EVENT']]
            all_entities.extend(entities)
            
            # Extract important keywords (nouns and adjectives)
            keywords = [token.lemma_.lower() for token in doc 
                       if token.pos_ in ['NOUN', 'ADJ'] and len(token.text) > 3 
                       and not token.is_stop]
            all_keywords.extend(keywords)
        
        # Get most common themes
        entity_counts = Counter(ent[0] for ent in all_entities)
        keyword_counts = Counter(all_keywords)
        
        return {
            'main_entities': dict(entity_counts.most_common(10)),
            'key_themes': dict(keyword_counts.most_common(10)),
            'entity_types': dict(Counter(ent[1] for ent in all_entities))
        }
   
    def _fallback_theme_analysis(self, texts):
        """Fallback theme analysis without spaCy"""
        tech_keywords = ['tech', 'review', 'smartphone', 'laptop', 'ai', 'software', 'hardware']
        gaming_keywords = ['game', 'gaming', 'play', 'stream', 'minecraft', 'fortnite']
        lifestyle_keywords = ['life', 'daily', 'routine', 'travel', 'fashion', 'beauty']
        education_keywords = ['learn', 'tutorial', 'explain', 'science', 'math', 'history']
        
        all_text = ' '.join(texts).lower()
        
        themes = {}
        for keyword in tech_keywords + gaming_keywords + lifestyle_keywords + education_keywords:
            count = all_text.count(keyword)
            if count > 0:
                themes[keyword] = count
            
        return {
            'main_entities': {},
            'key_themes': dict(Counter(themes).most_common(10)) if themes else {'general': 1},
            'entity_types': {}
        }
    
    def analyze_sentiment(self, texts):
        """Comprehensive sentiment analysis"""
        if not AI_FEATURES_AVAILABLE or not self.sentiment_analyzer:
            return self._fallback_sentiment(texts)
            
        sentiments = []
        emotions = {'positive': 0, 'neutral': 0, 'negative': 0}
        
        for text in texts:
            # VADER sentiment
            vader_scores = self.sentiment_analyzer.polarity_scores(text)
            
            # TextBlob sentiment
            try:
                blob = TextBlob(text)
                textblob_polarity = blob.sentiment.polarity
            except:
                textblob_polarity = 0
            
            # Combine both analyses
            combined_sentiment = {
                'vader_compound': vader_scores['compound'],
                'textblob_polarity': textblob_polarity,
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            }
            
            sentiments.append(combined_sentiment)
            
            # Categorize sentiment
            if vader_scores['compound'] >= 0.05:
                emotions['positive'] += 1
            elif vader_scores['compound'] <= -0.05:
                emotions['negative'] += 1
            else:
                emotions['neutral'] += 1
        
        # Calculate overall sentiment
        avg_vader = sum(s['vader_compound'] for s in sentiments) / len(sentiments) if sentiments else 0
        avg_textblob = sum(s['textblob_polarity'] for s in sentiments) / len(sentiments) if sentiments else 0
        
        return {
            'individual_sentiments': sentiments,
            'overall_sentiment': {
                'vader_avg': round(avg_vader, 3),
                'textblob_avg': round(avg_textblob, 3),
                'sentiment_distribution': emotions,
                'dominant_sentiment': max(emotions, key=emotions.get)
            }
        }
    
    def _fallback_sentiment(self, texts):
        """Fallback sentiment analysis without AI libraries"""
        positive_words = ['good', 'great', 'awesome', 'amazing', 'love', 'best', 'excellent', 'wonderful', 'fantastic', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'disappointing', 'annoying', 'frustrating', 'boring']
        
        emotions = {'positive': 0, 'neutral': 0, 'negative': 0}
        sentiments = []
        
        for text in texts:
            text_lower = text.lower()
            pos_count = sum(text_lower.count(word) for word in positive_words)
            neg_count = sum(text_lower.count(word) for word in negative_words)
            
            if pos_count > neg_count:
                sentiment_score = 0.3
                emotions['positive'] += 1
            elif neg_count > pos_count:
                sentiment_score = -0.3
                emotions['negative'] += 1
            else:
                sentiment_score = 0.0
                emotions['neutral'] += 1
            
            sentiments.append({
                'vader_compound': sentiment_score,
                'textblob_polarity': sentiment_score,
                'text_preview': text[:100] + '...' if len(text) > 100 else text
            })
        
        avg_sentiment = sum(s['vader_compound'] for s in sentiments) / len(sentiments) if sentiments else 0
        
        return {
            'individual_sentiments': sentiments,
            'overall_sentiment': {
                'vader_avg': round(avg_sentiment, 3),
                'textblob_avg': round(avg_sentiment, 3),
                'sentiment_distribution': emotions,
                'dominant_sentiment': max(emotions, key=emotions.get)
            }
        }
    
    def predict_trends(self, channel_data):
        """Predict trending topics and content patterns"""
        if not channel_data:
            return {
                'trend_direction': 'stable',
                'trend_strength': 0,
                'trending_topics': {'general': 1},
                'performance_pattern': {
                    'recent_avg_views': 0,
                    'high_engagement_count': 0,
                    'consistency_score': 0
                },
                'recommendations': ['Insufficient data for trend analysis']
            }
        
        recent_videos = []
        older_videos = []
        
        cutoff_date = datetime.now() - timedelta(days=30)
        
        for video in channel_data:
            try:
                video_date = datetime.fromisoformat(video['published_at'].replace('Z', '+00:00'))
                if video_date > cutoff_date:
                    recent_videos.append(video)
                else:
                    older_videos.append(video)
            except:
                recent_videos.append(video)  # If date parsing fails, assume recent
        
        # Initialize default values
        trend_direction = 'stable'
        trend_strength = 0
        recent_avg_views = 0
        older_avg_views = 0
        
        # Analyze recent vs older performance
        if recent_videos:
            recent_avg_views = sum(v.get('views', 0) for v in recent_videos) / len(recent_videos)
        
        if older_videos:
            older_avg_views = sum(v.get('views', 0) for v in older_videos) / len(older_videos)
        
        # Determine trend direction
        if recent_videos and older_videos and older_avg_views > 0:
            if recent_avg_views > older_avg_views:
                trend_direction = 'rising'
            elif recent_avg_views < older_avg_views:
                trend_direction = 'declining'
            else:
                trend_direction = 'stable'
            
            trend_strength = abs(recent_avg_views - older_avg_views) / older_avg_views
        
        # Extract trending topics from recent content
        recent_titles = [v.get('title', '') for v in recent_videos]
        trending_themes = self.analyze_content_themes(recent_titles)
        
        # Performance predictions
        high_performing_videos = [v for v in recent_videos if v.get('engagement_rate', 0) > 5]
        
        return {
            'trend_direction': trend_direction,
            'trend_strength': round(trend_strength, 3),
            'trending_topics': trending_themes['key_themes'],
            'performance_pattern': {
                'recent_avg_views': int(recent_avg_views),
                'high_engagement_count': len(high_performing_videos),
                'consistency_score': self._calculate_consistency_score(recent_videos)
            },
            'recommendations': self._generate_trend_recommendations(trend_direction, trending_themes)
        }
    
    def _calculate_consistency_score(self, videos):
        """Calculate how consistent the creator's performance is"""
        if not videos:
            return 0
            
        engagement_rates = [v.get('engagement_rate', 0) for v in videos if 'engagement_rate' in v]
        if not engagement_rates:
            return 0
            
        avg_engagement = sum(engagement_rates) / len(engagement_rates)
        variance = sum((x - avg_engagement) ** 2 for x in engagement_rates) / len(engagement_rates)
        
        # Lower variance = higher consistency (scale 0-10)
        consistency_score = max(0, 10 - (variance * 2))
        return round(consistency_score, 2)
    
    def _generate_trend_recommendations(self, trend_direction, themes):
        """Generate actionable recommendations based on trends"""
        recommendations = []
        
        if trend_direction == 'rising':
            recommendations.append("✅ Strong upward trajectory - Excellent partnership potential")
            recommendations.append("📈 Consider long-term collaboration agreements")
        elif trend_direction == 'declining':
            recommendations.append("⚠️ Performance declining - Monitor closely before committing")
            recommendations.append("🔍 Investigate content strategy changes")
        else:
            recommendations.append("📊 Stable performance - Reliable partnership option")
        
        # Theme-based recommendations
        key_themes = themes.get('key_themes', {})
        top_themes = list(key_themes.keys())[:3] if key_themes else []
        if top_themes:
            recommendations.append(f"🎯 Focus on trending topics: {', '.join(top_themes)}")
        
        return recommendations
    
    def parse_duration(self, duration):
        """Convert ISO 8601 duration to readable format"""
        try:
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
        except:
            pass
        return "0:00"

class CreatorProfiler:
    def __init__(self, ai_analyzer):
        self.ai_analyzer = ai_analyzer
    
    def create_creator_profile(self, channel_info, videos_data):
        """Create comprehensive creator personality and content profile"""
        if not videos_data:
            videos_data = []
            
        all_titles = [v.get('title', '') for v in videos_data]
        all_descriptions = [v.get('description', '') for v in videos_data]
        all_content = [text for text in all_titles + all_descriptions if text]
        
        if not all_content:
            all_content = ['No content available']
        
        # Content analysis
        themes = self.ai_analyzer.analyze_content_themes(all_content)
        sentiment = self.ai_analyzer.analyze_sentiment(all_content)
        trends = self.ai_analyzer.predict_trends(videos_data)
        
        # Performance metrics
        avg_views = sum(v.get('views', 0) for v in videos_data) / len(videos_data) if videos_data else 0
        avg_engagement = sum(v.get('engagement_rate', 0) for v in videos_data) / len(videos_data) if videos_data else 0
        
        # Content frequency analysis
        posting_frequency = self._analyze_posting_frequency(videos_data)
        
        # Brand safety assessment
        brand_safety = self._assess_brand_safety(all_content, sentiment)
        
        return {
            'creator_id': channel_info.get('id', 'unknown'),
            'creator_name': channel_info.get('title', 'Unknown Creator'),
            'profile_generated_at': datetime.now().isoformat(),
            'content_dna': {
                'primary_themes': themes.get('key_themes', {}),
                'content_entities': themes.get('main_entities', {}),
                'sentiment_profile': sentiment['overall_sentiment']
            },
            'performance_metrics': {
                'avg_views': int(avg_views),
                'avg_engagement_rate': round(avg_engagement, 2),
                'consistency_score': trends['performance_pattern']['consistency_score'],
                'trend_direction': trends['trend_direction']
            },
            'posting_behavior': posting_frequency,
            'brand_compatibility': {
                'safety_score': brand_safety['safety_score'],
                'risk_factors': brand_safety['risk_factors'],
                'opportunities': brand_safety['opportunities']
            },
            'recommendations': trends['recommendations']
        }
    
    def _analyze_posting_frequency(self, videos_data):
        """Analyze how frequently the creator posts"""
        if not videos_data:
            return {'frequency': 'unknown', 'consistency': 0}
            
        dates = []
        for video in videos_data:
            try:
                date = datetime.fromisoformat(video['published_at'].replace('Z', '+00:00'))
                dates.append(date)
            except:
                continue
        
        if len(dates) < 2:
            return {'frequency': 'insufficient_data', 'consistency': 0}
        
        dates.sort()
        intervals = []
        
        for i in range(1, len(dates)):
            interval = (dates[i] - dates[i-1]).days
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        
        # Determine frequency category
        if avg_interval <= 1:
            frequency = 'daily'
        elif avg_interval <= 3:
            frequency = 'multiple_per_week'
        elif avg_interval <= 7:
            frequency = 'weekly'
        elif avg_interval <= 14:
            frequency = 'bi_weekly'
        else:
            frequency = 'monthly_or_less'
        
        # Calculate consistency (lower variance in intervals = more consistent)
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals) if intervals else 0
        consistency = max(0, 100 - variance)
        
        return {
            'frequency': frequency,
            'avg_days_between_posts': round(avg_interval, 1),
            'consistency_percentage': round(consistency, 1)
        }
    
    def _assess_brand_safety(self, content_texts, sentiment_analysis):
        """Assess brand safety and partnership suitability"""
        risk_keywords = [
            'controversy', 'drama', 'exposed', 'scandal', 'beef', 'diss',
            'hate', 'toxic', 'cancel', 'problematic', 'inappropriate'
        ]
        
        opportunity_keywords = [
            'tutorial', 'review', 'educational', 'helpful', 'positive',
            'inspiring', 'motivational', 'family-friendly', 'professional'
        ]
        
        all_text = ' '.join(content_texts).lower()
        
        risk_count = sum(all_text.count(keyword) for keyword in risk_keywords)
        opportunity_count = sum(all_text.count(keyword) for keyword in opportunity_keywords)
        
        # Calculate safety score (1-10, higher is safer)
        base_safety_score = 8
        base_safety_score -= min(risk_count * 0.5, 4)  # Reduce for risk keywords
        base_safety_score += min(opportunity_count * 0.2, 2)  # Increase for positive keywords
        
        # Factor in sentiment
        overall_sentiment = sentiment_analysis['overall_sentiment']['vader_avg']
        if overall_sentiment < -0.1:
            base_safety_score -= 1
        elif overall_sentiment > 0.1:
            base_safety_score += 0.5
        
        safety_score = max(1, min(10, base_safety_score))
        
        # Generate specific feedback
        risk_factors = []
        opportunities = []
        
        if risk_count > 0:
            risk_factors.append(f"Contains {risk_count} potentially risky content references")
        if overall_sentiment < -0.1:
            risk_factors.append("Overall content sentiment leans negative")
        
        if opportunity_count > 0:
            opportunities.append(f"Contains {opportunity_count} brand-positive content elements")
        if overall_sentiment > 0.1:
            opportunities.append("Positive overall content sentiment")
        
        if not risk_factors:
            risk_factors.append("No significant risk factors identified")
        if not opportunities:
            opportunities.append("Standard partnership potential")
        
        return {
            'safety_score': round(safety_score, 1),
            'risk_factors': risk_factors,
            'opportunities': opportunities
        }

# Predefined channels list
MONITORED_CHANNELS = {
    "Tech": [
        {"name": "MKBHD", "handle": "mkbhd"},
        {"name": "Unbox Therapy", "handle": "unboxtherapy"},
        {"name": "Linus Tech Tips", "handle": "linustechtips"},
        {"name": "Dave2D", "handle": "dave2d"}
    ],
    "Gaming": [
        {"name": "PewDiePie", "handle": "pewdiepie"},
        {"name": "MrBeast Gaming", "handle": "mrbeastgaming"},
        {"name": "Jacksepticeye", "handle": "jacksepticeye"},
        {"name": "Markiplier", "handle": "markiplier"}
    ],
    "Lifestyle": [
        {"name": "Emma Chamberlain", "handle": "emmachamberlain"},
        {"name": "James Charles", "handle": "jamescharles"},
        {"name": "Dude Perfect", "handle": "dudeperfect"},
        {"name": "Casey Neistat", "handle": "caseyneistat"}
    ],
    "Education": [
        {"name": "Veritasium", "handle": "veritasium"},
        {"name": "Kurzgesagt", "handle": "kurzgesagt"},
        {"name": "Khan Academy", "handle": "khanacademy"},
        {"name": "Crash Course", "handle": "crashcourse"}
    ],
    "Business": [
        {"name": "Gary Vaynerchuk", "handle": "garyvee"},
        {"name": "Ali Abdaal", "handle": "aliabdaal"},
        {"name": "Peter McKinnon", "handle": "petermckinnon"},
        {"name": "Matt D'Avella", "handle": "mattdavella"}
    ]
}

class EnhancedYouTubeAnalyzer:
    def __init__(self):
        self.youtube = youtube
        self.ai_analyzer = AIAnalyzer()
        self.creator_profiler = CreatorProfiler(self.ai_analyzer)
        self.data_cache = {}
        self.setup_data_storage()
    
    def setup_data_storage(self):
        """Initialize JSON-based data storage"""
        try:
            os.makedirs('data', exist_ok=True)
            os.makedirs('data/cache', exist_ok=True)
            os.makedirs('data/profiles', exist_ok=True)
            os.makedirs('data/reports', exist_ok=True)
            
            # Initialize channels.json if it doesn't exist
            channels_file = 'data/channels.json'
            if not os.path.exists(channels_file):
                with open(channels_file, 'w') as f:
                    json.dump(MONITORED_CHANNELS, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not create data directories: {e}")
    
    def search_youtuber(self, query, max_results=10):
        """Enhanced YouTuber search with AI analysis"""
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
            
            if not channel_stats['items']:
                return {'error': 'Channel statistics not found'}
            
            channel_info = channel_stats['items'][0]
            
            # Get latest videos from this channel
            videos_data = self.get_channel_videos(channel_id, max_results)
            
            # Create AI-powered creator profile
            creator_profile = self.creator_profiler.create_creator_profile(
                {
                    'id': channel_id,
                    'title': channel_title,
                    'subscriber_count': int(channel_info['statistics'].get('subscriberCount', 0)),
                    'video_count': int(channel_info['statistics'].get('videoCount', 0)),
                    'view_count': int(channel_info['statistics'].get('viewCount', 0))
                },
                videos_data
            )
            
            # Cache the profile
            self.cache_creator_profile(channel_id, creator_profile)
            
            return {
                'channel_info': {
                    'id': channel_id,
                    'title': channel_title,
                    'subscriber_count': int(channel_info['statistics'].get('subscriberCount', 0)),
                    'video_count': int(channel_info['statistics'].get('videoCount', 0)),
                    'view_count': int(channel_info['statistics'].get('viewCount', 0)),
                    'thumbnail': channel_info['snippet']['thumbnails']['high']['url']
                },
                'videos': videos_data,
                'ai_analysis': creator_profile
            }
            
        except Exception as e:
            return {'error': f'YouTube API error: {str(e)}'}
    
    def get_channel_videos(self, channel_id, max_results=10):
        """Get latest videos with enhanced AI analysis"""
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
            videos_data.sort(key=lambda x: x.get('engagement_rate', 0), reverse=True)
            
            return videos_data
            
        except Exception as e:
            print(f"Error getting channel videos: {e}")
            return []
    
    def analyze_video(self, video):
        """Enhanced video analysis with AI insights"""
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
        try:
            publish_date = datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00'))
            days_since_publish = (datetime.now(publish_date.tzinfo) - publish_date).days
        except:
            days_since_publish = 0
        
        # Get video duration
        duration = self.ai_analyzer.parse_duration(video['contentDetails']['duration'])
        
        # Enhanced AI analysis
        content_text = f"{snippet['title']} {snippet.get('description', '')}"
        
        # Sentiment analysis
        sentiment_result = self.ai_analyzer.analyze_sentiment([content_text])
        
        # Theme analysis
        theme_result = self.ai_analyzer.analyze_content_themes([content_text])
        
        # Enhanced rule-based analysis
        analysis = self.enhanced_analysis(views, engagement_rate, days_since_publish, 
                                        snippet, sentiment_result, theme_result)
        
        return {
            'id': video['id'],
            'title': snippet['title'],
            'description': snippet.get('description', '')[:200] + '...' if len(snippet.get('description', '')) > 200 else snippet.get('description', ''),
            'thumbnail': snippet['thumbnails'].get('maxres', snippet['thumbnails'].get('high', snippet['thumbnails'].get('medium', {})))['url'],
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
    
    def enhanced_analysis(self, views, engagement_rate, days_since_publish, snippet, sentiment, themes):
        """Enhanced analysis incorporating AI insights"""
        # Basic viral score calculation
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
        
        # Recency factor
        if days_since_publish <= 7:
            viral_score += 2
        elif days_since_publish <= 30:
            viral_score += 1
        
        # AI-enhanced factors
        sentiment_score = sentiment['overall_sentiment']['vader_avg']
        if sentiment_score > 0.5:  # Very positive content
            viral_score += 1
        elif sentiment_score < -0.3:  # Negative content (controversial might be viral)
            viral_score += 0.5
        
        viral_score = min(viral_score, 10)
        
        # Enhanced content categorization using AI themes
        top_themes = list(themes.get('key_themes', {}).keys())[:3]
        content_category = self._determine_category_from_themes(snippet['title'], snippet.get('description', ''), top_themes)
        
        # Enhanced brand safety with AI sentiment
        brand_safety_score = 10
        risky_keywords = ['controversy', 'drama', 'exposed', 'scandal', 'beef', 'diss']
        title_lower = snippet['title'].lower()
        description_lower = snippet.get('description', '').lower()
        
        for keyword in risky_keywords:
            if keyword in title_lower or keyword in description_lower:
                brand_safety_score -= 2
        
        # Adjust based on sentiment
        if sentiment_score < -0.3:
            brand_safety_score -= 1
        elif sentiment_score > 0.3:
            brand_safety_score += 0.5
        
        brand_safety_score = max(1, min(10, brand_safety_score))
        
        # Enhanced engagement quality
        if engagement_rate > 8:
            engagement_quality = 'excellent'
        elif engagement_rate > 5:
            engagement_quality = 'high'
        elif engagement_rate > 2:
            engagement_quality = 'medium'
        else:
            engagement_quality = 'low'
        
        # AI-powered recommendations
        recommended_action = self._generate_ai_recommendation(viral_score, brand_safety_score, 
                                                           sentiment_score, engagement_rate, top_themes)
        
        return {
            'viral_score': viral_score,
            'content_category': content_category,
            'brand_safety_score': round(brand_safety_score, 1),
            'engagement_quality': engagement_quality,
            'recommended_action': recommended_action,
            'ai_insights': {
                'sentiment_score': round(sentiment_score, 3),
                'key_themes': top_themes,
                'content_tone': 'positive' if sentiment_score > 0.1 else 'negative' if sentiment_score < -0.1 else 'neutral'
            }
        }
    
    def _determine_category_from_themes(self, title, description, themes):
        """Use AI themes to determine content category"""
        text = f"{title} {description}".lower()
        
        # Enhanced category detection
        categories = {
            'Tech': ['tech', 'technology', 'review', 'smartphone', 'laptop', 'software', 'hardware', 'ai', 'gadget'],
            'Gaming': ['game', 'gaming', 'play', 'stream', 'minecraft', 'fortnite', 'gameplay'],
            'Entertainment': ['funny', 'comedy', 'react', 'challenge', 'prank', 'entertainment'],
            'Education': ['tutorial', 'learn', 'how to', 'explain', 'guide', 'education', 'science'],
            'Lifestyle': ['vlog', 'daily', 'routine', 'life', 'travel', 'lifestyle', 'fashion'],
            'Music': ['music', 'song', 'cover', 'sing', 'dance', 'album'],
            'News': ['news', 'update', 'breaking', 'report', 'current', 'politics']
        }
        
        # Check themes first
        for theme in themes:
            for category, keywords in categories.items():
                if theme.lower() in keywords:
                    return category
        
        # Fallback to keyword matching
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(text.count(keyword) for keyword in keywords)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        return 'General'
    
    def _generate_ai_recommendation(self, viral_score, brand_safety_score, sentiment_score, engagement_rate, themes):
        """Generate AI-powered partnership recommendations"""
        if viral_score >= 8 and brand_safety_score >= 8 and sentiment_score >= 0:
            return f"🎯 Excellent partnership opportunity! High viral potential with positive sentiment around {', '.join(themes[:2]) if themes else 'trending topics'}"
        elif viral_score >= 6 and brand_safety_score >= 7:
            return f"📈 Strong collaboration potential. Consider partnership for {', '.join(themes[:2]) if themes else 'current content themes'}"
        elif brand_safety_score < 6:
            return "⚠️ Potential brand safety concerns detected. Review content carefully before partnership"
        elif engagement_rate < 2:
            return "📊 Low engagement rates. Monitor performance before major investments"
        else:
            return f"🤔 Moderate potential. Good for niche campaigns targeting {', '.join(themes[:2]) if themes else 'specific audiences'}"
    
    def cache_creator_profile(self, channel_id, profile_data):
        """Cache creator profile data to JSON"""
        try:
            profile_file = f'data/profiles/{channel_id}.json'
            with open(profile_file, 'w') as f:
                json.dump(profile_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not cache profile: {e}")
    
    def get_all_monitored_channels(self):
        """Get list of all monitored channels"""
        try:
            with open('data/channels.json', 'r') as f:
                return json.load(f)
        except:
            return MONITORED_CHANNELS
    
    def analyze_multiple_creators(self, channel_queries, max_results_per_channel=5):
        """Analyze multiple creators for comparison"""
        results = {}
        
        for query in channel_queries:
            try:
                result = self.search_youtuber(query, max_results_per_channel)
                results[query] = result
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                results[query] = {'error': str(e)}
        
        return results
    
    def generate_trend_report(self, channel_data_list):
        """Generate comprehensive trend report across multiple channels"""
        all_videos = []
        all_themes = []
        performance_data = []
        
        for channel_data in channel_data_list:
            if 'error' in channel_data:
                continue
                
            if 'videos' in channel_data:
                all_videos.extend(channel_data['videos'])
            if 'ai_analysis' in channel_data:
                all_themes.extend(list(channel_data['ai_analysis']['content_dna']['primary_themes'].keys()))
                performance_data.append({
                    'name': channel_data['channel_info']['title'],
                    'avg_engagement': channel_data['ai_analysis']['performance_metrics']['avg_engagement_rate'],
                    'trend_direction': channel_data['ai_analysis']['performance_metrics']['trend_direction'],
                    'safety_score': channel_data['ai_analysis']['brand_compatibility']['safety_score']
                })
        
        # Cross-creator trend analysis
        theme_counter = Counter(all_themes)
        trending_topics = dict(theme_counter.most_common(10))
        
        # Performance benchmarks
        avg_engagement_benchmark = sum(p['avg_engagement'] for p in performance_data) / len(performance_data) if performance_data else 0
        
        # Generate insights
        insights = self._generate_trend_insights(trending_topics, performance_data, all_videos)
        
        return {
            'report_generated_at': datetime.now().isoformat(),
            'analyzed_channels_count': len(performance_data),
            'trending_topics_across_creators': trending_topics,
            'performance_benchmarks': {
                'avg_engagement_rate': round(avg_engagement_benchmark, 2),
                'high_performers': [p['name'] for p in performance_data if p['avg_engagement'] > avg_engagement_benchmark * 1.2],
                'rising_creators': [p['name'] for p in performance_data if p['trend_direction'] == 'rising']
            },
            'brand_safety_overview': {
                'safe_creators': [p['name'] for p in performance_data if p['safety_score'] >= 8],
                'moderate_risk': [p['name'] for p in performance_data if 6 <= p['safety_score'] < 8],
                'high_risk': [p['name'] for p in performance_data if p['safety_score'] < 6]
            },
            'actionable_insights': insights,
            'creator_comparison_data': performance_data
        }
    
    def _generate_trend_insights(self, trending_topics, performance_data, all_videos):
        """Generate actionable insights from trend data"""
        insights = []
        
        # Topic insights
        if trending_topics:
            top_topic = list(trending_topics.keys())[0]
            insights.append(f"🔥 '{top_topic}' is the hottest topic across creators - consider campaigns around this theme")
        
        # Performance insights
        high_performers = [p for p in performance_data if p['avg_engagement'] > 5]
        if high_performers:
            insights.append(f"⭐ {len(high_performers)} creators showing exceptional engagement rates")
        
        # Trend direction insights
        rising_count = len([p for p in performance_data if p['trend_direction'] == 'rising'])
        if rising_count > len(performance_data) * 0.5:
            insights.append("📈 Overall positive trend across monitored creators")
        
        # Content timing insights
        recent_videos = [v for v in all_videos if v['days_since_publish'] <= 7]
        if recent_videos:
            avg_recent_engagement = sum(v['engagement_rate'] for v in recent_videos) / len(recent_videos)
            insights.append(f"📊 Recent content averaging {avg_recent_engagement:.1f}% engagement")
        
        # Safety insights
        safe_creators = [p for p in performance_data if p['safety_score'] >= 8]
        insights.append(f"🛡️ {len(safe_creators)} creators classified as brand-safe for partnerships")
        
        if not insights:
            insights.append("📊 Baseline analysis completed - monitor for emerging trends")
        
        return insights

# Initialize enhanced analyzer
enhanced_analyzer = EnhancedYouTubeAnalyzer()

# Initialize scheduler for real-time monitoring
if SCHEDULER_AVAILABLE:
    scheduler = BackgroundScheduler()
    scheduler.start()

    def background_channel_update():
        """Background task to update all monitored channels"""
        print("🔄 Running background update for all monitored channels...")
        try:
            channels = enhanced_analyzer.get_all_monitored_channels()
            
            for category, channel_list in channels.items():
                for channel in channel_list[:2]:  # Limit to first 2 per category for API limits
                    try:
                        result = enhanced_analyzer.search_youtuber(channel['name'], 5)
                        if 'error' not in result:
                            print(f"✅ Updated {channel['name']}")
                        time.sleep(2)  # Rate limiting
                    except Exception as e:
                        print(f"❌ Error updating {channel['name']}: {e}")
            
            # Cache timestamp
            try:
                with open('data/last_update.json', 'w') as f:
                    json.dump({'last_update': datetime.now().isoformat()}, f)
            except Exception as e:
                print(f"Warning: Could not save timestamp: {e}")
                
            print("✅ Background update completed")
        except Exception as e:
            print(f"❌ Background update failed: {e}")

    # Schedule background updates every 2 hours
    scheduler.add_job(
        func=background_channel_update,
        trigger="interval",
        hours=2,
        id='channel_update'
    )

# Flask Routes

@app.route('/')
def index():
    """Serve the enhanced HTML dashboard"""
    # Try to read the HTML file from the same directory
    try:
        with open('dashboard.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        # Return a basic HTML page if dashboard.html is not found
        return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>AI YouTube Analytics</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        .container { max-width: 800px; margin: 0 auto; text-align: center; }
        input { padding: 10px; margin: 10px; width: 300px; border: none; border-radius: 5px; }
        button { padding: 10px 20px; background: #4ECDC4; color: white; border: none; border-radius: 5px; cursor: pointer; }
        .results { background: white; color: black; padding: 20px; margin: 20px 0; border-radius: 10px; display: none; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI YouTube Analytics</h1>
        <input type="text" id="search" placeholder="Enter YouTuber name" />
        <button onclick="searchYouTuber()">Analyze</button>
        <div id="results" class="results"></div>
    </div>
    <script>
        async function searchYouTuber() {
            const query = document.getElementById('search').value;
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            const data = await response.json();
            document.getElementById('results').innerHTML = JSON.stringify(data, null, 2);
            document.getElementById('results').style.display = 'block';
        }
    </script>
</body>
</html>
        """)

@app.route('/api/search', methods=['POST'])
def search_youtuber():
    """Enhanced API endpoint with AI analysis"""
    try:
        data = request.json
        query = data.get('query', '').strip()
        max_results = data.get('max_results', 10)
        
        if not query:
            return jsonify({'error': 'Please provide a search query'}), 400
        
        # Analyze YouTuber with AI enhancements
        result = enhanced_analyzer.search_youtuber(query, max_results)
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/api/channels')
def get_all_channels():
    """Get list of all monitored channels"""
    try:
        channels = enhanced_analyzer.get_all_monitored_channels()
        return jsonify(channels)
    except Exception as e:
        return jsonify({'error': f'Failed to load channels: {str(e)}'}), 500

@app.route('/api/compare', methods=['POST'])
def compare_creators():
    """Compare multiple creators"""
    try:
        data = request.json
        channel_queries = data.get('channels', [])
        
        if not channel_queries:
            return jsonify({'error': 'Please provide channel queries to compare'}), 400
        
        if len(channel_queries) > 5:
            return jsonify({'error': 'Maximum 5 channels can be compared at once'}), 400
        
        results = enhanced_analyzer.analyze_multiple_creators(channel_queries)
        
        # Generate comparison report
        valid_results = [r for r in results.values() if 'error' not in r]
        if valid_results:
            trend_report = enhanced_analyzer.generate_trend_report(valid_results)
            return jsonify({
                'individual_results': results,
                'comparison_report': trend_report
            })
        else:
            return jsonify({'error': 'No valid channels found for comparison'}), 400
    except Exception as e:
        return jsonify({'error': f'Comparison failed: {str(e)}'}), 500

@app.route('/api/trending')
def get_trending_insights():
    """Get trending insights across all monitored channels"""
    try:
        # Get real trending data from actual channels
        channels = enhanced_analyzer.get_all_monitored_channels()
        sample_channels = []
        
        # Get sample from each category
        for category, channel_list in channels.items():
            if channel_list:
                sample_channels.append(channel_list[0]['name'])
        
        if sample_channels:
            results = enhanced_analyzer.analyze_multiple_creators(sample_channels[:5])
            valid_results = [r for r in results.values() if 'error' not in r]
            
            if valid_results:
                trend_report = enhanced_analyzer.generate_trend_report(valid_results)
                return jsonify(trend_report)
        
        # Fallback data
        insights = {
            'trending_topics_across_creators': {'AI': 5, 'Gaming': 4, 'Tech': 3, 'Review': 3, 'Tutorial': 2},
            'performance_benchmarks': {
                'avg_engagement_rate': 4.2,
                'high_performers': ['Sample Channel'],
                'rising_creators': ['Rising Creator']
            },
            'last_analysis': datetime.now().isoformat()
        }
        
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': f'Failed to load trending data: {str(e)}'}), 500

@app.route('/api/report/generate', methods=['POST'])
def generate_executive_report():
    """Generate executive summary report"""
    try:
        data = request.json
        selected_channels = data.get('channels', [])
        report_type = data.get('type', 'standard')
        
        if not selected_channels:
            return jsonify({'error': 'Please select channels for the report'}), 400
        
        # Analyze selected channels
        results = enhanced_analyzer.analyze_multiple_creators(selected_channels)
        valid_results = [r for r in results.values() if 'error' not in r]
        
        if not valid_results:
            return jsonify({'error': 'No valid data found for selected channels'}), 400
        
        # Generate comprehensive report
        trend_report = enhanced_analyzer.generate_trend_report(valid_results)
        
        # Create executive summary
        executive_summary = {
            'report_title': f'{report_type.title()} Influencer Analysis Report',
            'generated_at': datetime.now().isoformat(),
            'channels_analyzed': len(valid_results),
            'key_findings': trend_report['actionable_insights'],
            'recommendations': {
                'top_partnership_candidates': trend_report['performance_benchmarks']['high_performers'][:3],
                'trending_content_themes': list(trend_report['trending_topics_across_creators'].keys())[:5],
                'brand_safety_assessment': trend_report['brand_safety_overview']
            },
            'next_steps': [
                'Monitor high-performing creators for partnership opportunities',
                'Develop content strategy around trending topics',
                'Schedule follow-up analysis in 48 hours'
            ]
        }
        
        # Cache report
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(f'data/reports/{report_filename}', 'w') as f:
                json.dump({
                    'executive_summary': executive_summary,
                    'detailed_analysis': trend_report,
                    'raw_data': valid_results
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save report: {e}")
        
        return jsonify({
            'executive_summary': executive_summary,
            'detailed_analysis': trend_report,
            'report_id': report_filename
        })
        
    except Exception as e:
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.route('/api/status')
def get_system_status():
    """Get system status and last update info"""
    try:
        # Check last update time
        try:
            with open('data/last_update.json', 'r') as f:
                last_update_data = json.load(f)
            last_update = last_update_data['last_update']
        except:
            last_update = "Never"
        
        # Count cached profiles
        try:
            profile_count = len([f for f in os.listdir('data/profiles') if f.endswith('.json')])
        except:
            profile_count = 0
        
        status = {
            'system_status': 'operational',
            'last_background_update': last_update,
            'cached_profiles': profile_count,
            'monitored_channels': sum(len(channels) for channels in MONITORED_CHANNELS.values()),
            'ai_features_active': AI_FEATURES_AVAILABLE,
            'scheduler_running': SCHEDULER_AVAILABLE and scheduler.running if SCHEDULER_AVAILABLE else False
        }
        
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': f'Status check failed: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Enhanced AI-Powered YouTube Analytics Dashboard")
    print("🤖 AI Features:", "✅ Active" if AI_FEATURES_AVAILABLE else "❌ Not Available")
    print("📊 Real-time Monitoring:", "✅ Active" if SCHEDULER_AVAILABLE else "❌ Not Available")
    print("📡 Server starting on: http://localhost:5000")
    
    if not AI_FEATURES_AVAILABLE:
        print("\n📦 To enable full AI features, install:")
        print("pip install vaderSentiment textblob spacy")
        print("python -m spacy download en_core_web_sm")
    
    if not SCHEDULER_AVAILABLE:
        print("📦 To enable background monitoring, install:")
        print("pip install apscheduler")
    
    if YOUTUBE_API_KEY == 'AIzaSyAZGZp3p2VRWcczLzgqF6FigM93iaf9CyU':
        print("\n⚠️  Please replace the YouTube API key with your own!")
        print("Get one at: https://console.developers.google.com/")
    
    print("\n" + "="*60)
    print("Dashboard Features:")
    print("✅ Individual Creator Search & Analysis")
    print("✅ Multi-Creator Comparison")
    print("✅ Brand Safety Assessment")
    print("✅ Executive Report Generation")
    print("✅ Real-time Performance Tracking")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')
# AI-Agent-Hackathon

# AI-Powered YouTube Analytics Dashboard

## Quick Setup Guide

### Issues Fixed:
1. ✅ **API Key Configuration** - Now uses environment variables
2. ✅ **Error Handling** - Better error messages and fallbacks
3. ✅ **Demo Mode** - Works without API key for testing
4. ✅ **Frontend-Backend Communication** - Fixed CORS and JSON handling
5. ✅ **File Structure** - Proper HTML file loading

### Prerequisites

1. **Python 3.7+** installed
2. **pip** package manager
3. **YouTube Data API v3 Key** (optional for demo mode)

### Step 1: Install Dependencies

```bash
# Core dependencies
pip install flask flask-cors google-api-python-client

# AI features (optional but recommended)
pip install vaderSentiment textblob spacy

# Background scheduling (optional)
pip install apscheduler

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### Step 2: Get YouTube API Key (Optional)

1. Go to [Google Cloud Console](https://console.developers.google.com/)
2. Create a new project or select existing one
3. Enable **YouTube Data API v3**
4. Create credentials (API Key)
5. Copy your API key

### Step 3: Set Environment Variable

**Windows:**
```cmd
set YOUTUBE_API_KEY=your_api_key_here
```

**Mac/Linux:**
```bash
export YOUTUBE_API_KEY=your_api_key_here
```

**Or create a `.env` file:**
```
YOUTUBE_API_KEY=your_api_key_here
```

### Step 4: Create Files

1. **Save the backend code** as `app.py`
2. **Save the frontend code** as `dashboard.html` (same directory as app.py)

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Access Dashboard

Open your browser and go to:
```
http://localhost:5000
```

## Features

### ✅ Working Features:
- **Creator Search** - Search and analyze any YouTuber
- **AI Analysis** - Sentiment analysis and content themes
- **Performance Metrics** - Engagement rates and trends
- **Brand Safety Assessment** - Risk evaluation
- **Demo Mode** - Works without API key
- **Comparison Tool** - Compare multiple creators
- **Trending Insights** - Platform-wide trends

### 🔧 Demo Mode vs Live Mode:

| Feature | Demo Mode | Live Mode |
|---------|-----------|-----------|
| Creator Search | ✅ Sample data | ✅ Real YouTube data |
| AI Analysis | ✅ Simulated | ✅ Real analysis |
| Performance Metrics | ✅ Demo metrics | ✅ Actual metrics |
| Video Analysis | ✅ Sample videos | ✅ Latest videos |

## Troubleshooting

### Issue: "No channels found" or API errors
**Solution:** Check if your API key is set correctly:
```bash
echo $YOUTUBE_API_KEY  # Mac/Linux
echo %YOUTUBE_API_KEY% # Windows
```

### Issue: Charts not displaying
**Solution:** Ensure Chart.js is loading from CDN. Check browser console for errors.

### Issue: AI features not working
**Solution:** Install AI dependencies:
```bash
pip install vaderSentiment textblob spacy
python -m spacy download en_core_web_sm
```

### Issue: Dashboard shows setup page
**Solution:** Ensure `dashboard.html` is in the same directory as `app.py`

### Issue: CORS errors
**Solution:** The backend includes CORS headers. If issues persist, try:
```python
CORS(app, origins=["http://localhost:5000"])
```

## Testing the System

### 1. Test Search Function
Try searching for popular YouTubers:
- "MrBeast"
- "MKBHD" 
- "PewDiePie"
- "Kurzgesagt"

### 2. Test Comparison
Compare multiple creators:
```
MrBeast, PewDiePie, MKBHD
```

### 3. Check System Status
The status banner should show:
- 🟢 Green: Live YouTube data
- 🔄 Yellow: Demo mode
- 🔴 Red: Connection issues

## API Endpoints

### POST `/api/search`
```json
{
  "query": "MrBeast",
  "max_results": 10
}
```

### POST `/api/compare`
```json
{
  "channels": ["MrBeast", "PewDiePie", "MKBHD"]
}
```

### GET `/api/status`
Returns system status and configuration

### GET `/api/trending`
Returns trending insights across creators

## File Structure

```
project/
├── app.py              # Backend Flask application
├── dashboard.html      # Frontend dashboard
├── data/              # Auto-created for caching
│   ├── cache/         # API response cache
│   ├── profiles/      # Creator profiles
│   └── reports/       # Generated reports
└── README.md          # This guide
```

## Next Steps

1. **Get your YouTube API key** for live data
2. **Install all AI dependencies** for full analysis
3. **Customize the dashboard** for your brand needs
4. **Set up automated monitoring** for regular updates

## Pro Tips

1. **Rate Limiting:** The system includes built-in rate limiting for API calls
2. **Caching:** Creator profiles are cached to reduce API usage
3. **Error Recovery:** System falls back to demo data if API fails
4. **Responsive Design:** Dashboard works on mobile and desktop
5. **Real-time Updates:** Status banner shows current system state

## Support

If you encounter issues:
1. Check the console output for error messages
2. Verify your API key is set correctly
3. Ensure all dependencies are installed
4. Try demo mode first to test functionality

---

**🎯 Your AI-powered influencer analytics dashboard is ready!**

Start with demo mode to explore features, then add your API key for live YouTube data analysis.

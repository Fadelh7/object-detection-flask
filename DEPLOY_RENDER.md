# Deploy to Render (Free Tier)

## Prerequisites
1. GitHub account with your code pushed
2. Render account (free at https://render.com)

## Step-by-Step Deployment

### 1. Push Your Code to GitHub
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. Connect to Render
1. Go to https://render.com and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select this repository: `object-detection-flask`

### 3. Configure the Service
- **Name**: `object-detection-flask` (or your preferred name)
- **Branch**: `main`
- **Runtime**: `Docker`
- **Build Command**: (leave empty - uses Dockerfile)
- **Start Command**: (leave empty - uses Dockerfile CMD)

### 4. Environment Variables
Render will auto-detect from `render.yaml`, but you can also set manually:
- `PORT`: `8080`
- `FLASK_ENV`: `production`
- `YOLO_MODEL_PATH`: `models/best.pt`
- `CONF_THRES`: `0.25`

### 5. Deploy
1. Click "Create Web Service"
2. Render will automatically build and deploy your Docker container
3. This may take 10-15 minutes for the first build

### 6. Access Your App
- Your app will be available at: `https://your-service-name.onrender.com`
- Health check: `https://your-service-name.onrender.com/health`

## Important Notes for Free Tier

### Limitations:
- **Sleep Mode**: App goes to sleep after 15 minutes of inactivity
- **Cold Start**: First request after sleep takes ~30-60 seconds
- **Build Time**: Limited to 500 build minutes/month
- **Memory**: 512MB RAM limit
- **CPU**: Shared CPU

### Optimizations for Free Tier:
- Single Gunicorn worker to save memory
- CPU-only PyTorch (no GPU needed)
- Efficient model loading
- Health check endpoint for monitoring

### Model Size Considerations:
- Your `best.pt` model (~6MB) is fine for free tier
- Total app size ~2.5GB fits within limits
- Consider model optimization if you hit memory issues

## Troubleshooting

### If Build Fails:
1. Check build logs in Render dashboard
2. Ensure `models/best.pt` is committed to Git
3. Verify Dockerfile builds locally first

### If App Crashes:
1. Check application logs in Render dashboard
2. Usually memory-related on free tier
3. Consider reducing model complexity

### Performance Tips:
1. Keep the app "warm" with periodic requests
2. Use caching for frequently accessed images
3. Optimize image sizes before upload

## Monitoring
- Use Render's built-in monitoring
- Health check endpoint: `/health`
- View logs in Render dashboard

## Cost
- **Free tier**: $0/month
- **Upgrade options**: If you need persistent storage or more resources

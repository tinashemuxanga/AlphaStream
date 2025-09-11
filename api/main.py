"""
AlphaStream API Module

FastAPI endpoints for model inference, signal generation,
and real-time streaming via WebSockets.
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
import asyncio
import redis.asyncio as redis
from contextlib import asynccontextmanager
import joblib
import logging
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Internal imports
from ml.models import ModelFactory
from ml.dataset import DataLoader, DatasetBuilder
from ml.features import FeatureEngineer, TargetGenerator
from backtesting.engine import BacktestEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions."""
    symbol: str
    features: Optional[Dict[str, float]] = None
    lookback_days: int = Field(default=30, description="Days of historical data to use")
    model_type: str = Field(default="ensemble", description="Model type to use")
    
class SignalRequest(BaseModel):
    """Request model for trading signals."""
    symbols: List[str]
    signal_type: str = Field(default="classification", description="Signal type")
    threshold: float = Field(default=0.6, description="Confidence threshold")
    
class BacktestRequest(BaseModel):
    """Request model for backtesting."""
    symbol: str
    start_date: str
    end_date: str
    model_type: str = Field(default="xgboost")
    initial_capital: float = Field(default=100000)
    position_size: str = Field(default="equal_weight")
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
class ModelUpdateRequest(BaseModel):
    """Request model for model updates."""
    model_type: str
    retrain: bool = Field(default=False)
    symbols: List[str]
    start_date: str
    end_date: str


# Global variables for model management
models = {}
feature_engineer = None
redis_client = None
active_websockets = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global models, feature_engineer, redis_client
    
    logger.info("Starting AlphaStream API...")
    
    # Initialize Redis
    try:
        redis_client = await redis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True
        )
        await redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Running without cache.")
        redis_client = None
    
    # Load models
    model_dir = Path("models")
    if model_dir.exists():
        for model_file in model_dir.glob("*.pkl"):
            try:
                model_name = model_file.stem
                models[model_name] = joblib.load(model_file)
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_file}: {e}")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    logger.info("Feature engineer initialized")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AlphaStream API...")
    if redis_client:
        await redis_client.close()
    
    # Close all websockets
    for ws in active_websockets:
        await ws.close()


# Initialize FastAPI app
app = FastAPI(
    title="AlphaStream API",
    description="ML-powered trading signal generation API",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models),
        "redis_connected": redis_client is not None
    }


# Prediction endpoints
@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Generate prediction for a symbol.
    
    Returns prediction, confidence, and suggested action.
    """
    try:
        # Check if model exists
        if request.model_type not in models:
            # Try to load default model
            if "ensemble" in models:
                model = models["ensemble"]
            else:
                raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        else:
            model = models[request.model_type]
        
        # Get or generate features
        if request.features:
            # Use provided features
            features_df = pd.DataFrame([request.features])
        else:
            # Load historical data and generate features
            data_loader = DataLoader()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=request.lookback_days + 100)  # Extra for feature calculation
            
            data = data_loader.load(
                request.symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if data.empty:
                raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
            
            # Generate features
            features_df = feature_engineer.transform(data)
            features_df = features_df.iloc[-1:]  # Get latest features
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        
        # Get prediction probability if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features_df)[0]
            confidence = float(max(proba))
        
        # Determine action
        if prediction == 1:
            action = "BUY"
        elif prediction == -1:
            action = "SELL"
        else:
            action = "HOLD"
        
        # Cache result in Redis
        if redis_client:
            cache_key = f"prediction:{request.symbol}:{request.model_type}"
            cache_value = json.dumps({
                "prediction": int(prediction),
                "confidence": confidence,
                "action": action,
                "timestamp": datetime.now().isoformat()
            })
            await redis_client.setex(cache_key, 300, cache_value)  # Cache for 5 minutes
        
        return {
            "symbol": request.symbol,
            "prediction": int(prediction),
            "confidence": confidence,
            "action": action,
            "model_type": request.model_type,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals")
async def generate_signals(request: SignalRequest):
    """
    Generate trading signals for multiple symbols.
    
    Returns ranked list of signals with confidence scores.
    """
    try:
        signals = []
        
        for symbol in request.symbols:
            # Check cache first
            if redis_client:
                cache_key = f"signal:{symbol}"
                cached = await redis_client.get(cache_key)
                if cached:
                    signals.append(json.loads(cached))
                    continue
            
            # Generate prediction
            pred_request = PredictionRequest(
                symbol=symbol,
                model_type="ensemble"
            )
            
            try:
                result = await predict(pred_request)
                
                # Filter by threshold
                if result["confidence"] and result["confidence"] >= request.threshold:
                    signal = {
                        "symbol": symbol,
                        "signal": result["action"],
                        "strength": result["confidence"],
                        "timestamp": result["timestamp"]
                    }
                    signals.append(signal)
                    
                    # Cache signal
                    if redis_client:
                        await redis_client.setex(
                            f"signal:{symbol}",
                            300,
                            json.dumps(signal)
                        )
            except Exception as e:
                logger.warning(f"Failed to generate signal for {symbol}: {e}")
                continue
        
        # Sort by signal strength
        signals.sort(key=lambda x: x.get("strength", 0), reverse=True)
        
        return {
            "signals": signals,
            "total": len(signals),
            "threshold": request.threshold,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Signal generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """
    Run backtest for a strategy.
    
    Returns performance metrics and equity curve.
    """
    try:
        # Load data
        data_loader = DataLoader()
        data = data_loader.load(
            request.symbol,
            request.start_date,
            request.end_date
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {request.symbol}")
        
        # Generate features
        features = feature_engineer.transform(data)
        
        # Get model
        if request.model_type not in models:
            raise HTTPException(status_code=404, detail=f"Model {request.model_type} not found")
        
        model = models[request.model_type]
        
        # Generate predictions
        predictions = model.predict(features)
        
        # Convert to signals
        signals = pd.Series(predictions, index=features.index)
        
        # Run backtest
        engine = BacktestEngine(
            data,
            initial_capital=request.initial_capital,
            position_size_method=request.position_size
        )
        
        engine.add_signals(signals)
        results = engine.run(
            stop_loss=request.stop_loss,
            take_profit=request.take_profit
        )
        
        # Prepare response
        metrics = results['metrics']
        equity_curve = results['equity_curve'].to_dict()
        
        return {
            "symbol": request.symbol,
            "period": f"{request.start_date} to {request.end_date}",
            "metrics": {
                "total_return": metrics['total_return'],
                "sharpe_ratio": metrics['sharpe_ratio'],
                "max_drawdown": metrics['max_drawdown'],
                "win_rate": metrics['win_rate'],
                "total_trades": metrics['total_trades']
            },
            "equity_curve": equity_curve,
            "final_capital": float(results['equity_curve'].iloc[-1]),
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket for real-time streaming
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time signal streaming.
    
    Clients can subscribe to symbols and receive real-time updates.
    """
    await websocket.accept()
    active_websockets.add(websocket)
    
    subscribed_symbols = set()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("action") == "subscribe":
                symbols = data.get("symbols", [])
                subscribed_symbols.update(symbols)
                await websocket.send_json({
                    "type": "subscription",
                    "symbols": list(subscribed_symbols),
                    "status": "subscribed"
                })
                
            elif data.get("action") == "unsubscribe":
                symbols = data.get("symbols", [])
                subscribed_symbols.difference_update(symbols)
                await websocket.send_json({
                    "type": "subscription",
                    "symbols": list(subscribed_symbols),
                    "status": "unsubscribed"
                })
            
            # Send real-time updates
            if subscribed_symbols:
                for symbol in subscribed_symbols:
                    try:
                        # Generate prediction
                        pred_request = PredictionRequest(symbol=symbol)
                        result = await predict(pred_request)
                        
                        # Send to client
                        await websocket.send_json({
                            "type": "signal",
                            "data": result
                        })
                        
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "symbol": symbol,
                            "error": str(e)
                        })
                
                # Wait before next update
                await asyncio.sleep(5)  # Update every 5 seconds
                
    except WebSocketDisconnect:
        active_websockets.remove(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        active_websockets.remove(websocket)
        await websocket.close()


@app.post("/models/update")
async def update_model(request: ModelUpdateRequest, background_tasks: BackgroundTasks):
    """
    Update or retrain a model.
    
    This endpoint triggers model retraining in the background.
    """
    try:
        # Add background task for model training
        background_tasks.add_task(
            retrain_model,
            request.model_type,
            request.symbols,
            request.start_date,
            request.end_date
        )
        
        return {
            "status": "training_started",
            "model_type": request.model_type,
            "message": "Model training initiated in background",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def retrain_model(model_type: str, symbols: List[str], start_date: str, end_date: str):
    """Background task to retrain a model."""
    try:
        logger.info(f"Starting retraining for {model_type}")
        
        # Load data
        data_loader = DataLoader()
        all_data = []
        
        for symbol in symbols:
            data = data_loader.load(symbol, start_date, end_date)
            if not data.empty:
                data['symbol'] = symbol
                all_data.append(data)
        
        if not all_data:
            logger.error("No data available for training")
            return
        
        combined_data = pd.concat(all_data)
        
        # Build dataset
        dataset_builder = DatasetBuilder()
        features, targets = dataset_builder.build(combined_data)
        
        # Split data
        split_idx = int(len(features) * 0.8)
        X_train = features.iloc[:split_idx]
        y_train = targets.iloc[:split_idx]
        X_val = features.iloc[split_idx:]
        y_val = targets.iloc[split_idx:]
        
        # Train model
        model = ModelFactory.create(model_type, "classification")
        model.train(X_train, y_train, (X_val, y_val))
        
        # Save model
        model_path = Path("models") / f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        model_path.parent.mkdir(exist_ok=True)
        joblib.dump(model, model_path)
        
        # Update global models
        models[model_type] = model
        
        logger.info(f"Model {model_type} retrained successfully")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {e}")


@app.get("/models")
async def list_models():
    """List available models."""
    return {
        "models": list(models.keys()),
        "total": len(models),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/features/importance/{model_type}")
async def get_feature_importance(model_type: str):
    """Get feature importance for a model."""
    try:
        if model_type not in models:
            raise HTTPException(status_code=404, detail=f"Model {model_type} not found")
        
        model = models[model_type]
        
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            
            # Get feature names
            feature_names = feature_engineer.get_feature_names() if feature_engineer else []
            
            # Create importance dict
            importance_dict = {}
            for i, imp in enumerate(importance[:len(feature_names)]):
                importance_dict[feature_names[i]] = float(imp)
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )[:50])  # Top 50 features
            
            return {
                "model_type": model_type,
                "feature_importance": sorted_importance,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=400, detail="Model does not support feature importance")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feature importance error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

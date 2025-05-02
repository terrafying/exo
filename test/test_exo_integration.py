import pytest
import requests
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator
import uvicorn
from exo.main import app, shutdown_event, node, run
from exo.inference.inference_engine import get_inference_engine
from exo.download.shard_download import NoopShardDownloader
from exo.networking.udp.udp_discovery import UDPDiscovery
from exo.networking.grpc.grpc_peer_handle import GRPCPeerHandle
from exo.topology.ring_memory_weighted_partitioning_strategy import RingMemoryWeightedPartitioningStrategy
from exo.orchestration.node import Node
import time
import socket
import contextlib
from fastapi.testclient import TestClient

TEST_HOST = "127.0.0.1"
TEST_PORT = 8765
BASE_URL = f"http://{TEST_HOST}:{TEST_PORT}"

def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((TEST_HOST, port)) == 0

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def exo_server():
    """Start the exo server as a background task."""
    # Initialize the node with test configuration
    global node
    shard_downloader = NoopShardDownloader()
    inference_engine = get_inference_engine("dummy", shard_downloader)
    discovery = UDPDiscovery(
        "test-node",
        5678,
        5678,
        5678,
        lambda peer_id, address, description, device_capabilities: GRPCPeerHandle(peer_id, address, description, device_capabilities),
        discovery_timeout=1
    )
    node = Node(
        "test-node",
        None,
        inference_engine,
        discovery,
        shard_downloader,
        partitioning_strategy=RingMemoryWeightedPartitioningStrategy(),
        max_generate_tokens=1000,
        default_sample_temperature=0.7
    )
    
    # Create a test client
    client = TestClient(app)
    
    yield client
    
    # Cleanup
    shutdown_event.set()

@pytest.mark.asyncio
async def test_model_search(exo_server):
    """Test model search functionality."""
    response = exo_server.post(
        "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "non-existent-model",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        }
    )
    assert response.status_code == 404

@pytest.mark.asyncio
async def test_model_execution(exo_server):
    """Test model execution."""
    response = exo_server.post(
        "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "deepseek-prover-v2",
            "messages": [{
                "role": "user",
                "content": "Say hello!"
            }],
            "temperature": 0.7
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]

@pytest.mark.asyncio
async def test_search_and_execute(exo_server):
    """Test combined search and execution."""
    response = exo_server.post(
        "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "deepseek-prover-v2",
            "messages": [{
                "role": "user",
                "content": "Say hello!"
            }],
            "temperature": 0.7
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "message" in data["choices"][0]
    assert "content" in data["choices"][0]["message"]

@pytest.mark.asyncio
async def test_model_validation(exo_server):
    """Test model validation and error handling."""
    
    # Test case 1: Invalid model name
    response = exo_server.post(
        "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "non-existent-model",
            "messages": [{
                "role": "user",
                "content": "Hello"
            }]
        }
    )
    assert response.status_code == 404
    
    # Test case 2: Missing required fields
    response = exo_server.post(
        "/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "deepseek-prover-v2"
        }
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_concurrent_requests(exo_server):
    """Test handling of concurrent requests."""
    
    def make_request():
        return exo_server.post(
            "/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": "deepseek-prover-v2",
                "messages": [{
                    "role": "user",
                    "content": "Say hello!"
                }],
                "temperature": 0.7
            }
        )
    
    # Make 5 concurrent requests
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request) for _ in range(5)]
        responses = [f.result() for f in futures]
    
    # Verify all responses
    for response in responses:
        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"] 
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# Copyright (c) 2026 BeingBeyond Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0



from typing import Any, Dict, Optional
from .service import BaseInferenceClient, BaseInferenceServer


# ==============================================================================
# Server
# ==============================================================================

class BeingHInferenceServer(BaseInferenceServer):
    """
    Inference server for Being-H VLA policy.
    
    Exposes endpoints for:
    - get_action: Action inference from observations
    - get_modality_config: Query policy modality configuration
    """

    def __init__(
        self,
        policy: Any,
        host: str = "*",
        port: int = 5555,
        api_token: Optional[str] = None
    ):
        """
        Initialize inference server.
        
        Args:
            policy: Policy object with get_action() and get_modality_config() methods
            host: Host address to bind (default: "*" for all interfaces)
            port: Port to listen on (default: 5555)
            api_token: Optional API token for authentication
            
        Example:
            >>> from BeingH.inference.policy import InternVLPolicy
            >>> policy = InternVLPolicy(model_path="path/to/model", ...)
            >>> server = BeingHInferenceServer(policy, port=5555)
            >>> server.start()
        """

        super().__init__(host, port, api_token)

        # Register policy endpoints
        self.register_endpoint("get_action", policy.get_action)
        self.register_endpoint("get_modality_config", policy.get_modality_config, requires_input=False)


# ==============================================================================
# Client
# ==============================================================================

class BeingHInferenceClient(BaseInferenceClient):
    """
    Client for calling remote Being-H inference server.
    
    Provides methods to:
    - Query action predictions from observations
    - Retrieve policy configuration
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5555,
        api_token: Optional[str] = None
    ):
        """
        Initialize inference client.
        
        Args:
            host: Server hostname (default: "localhost")
            port: Server port (default: 5555)
            api_token: Optional API token for authentication
            
        Example:
            >>> client = BeingHInferenceClient(host="192.168.1.100", port=5555)
            >>> actions = client.get_action(observations)
        """
        super().__init__(host=host, port=port, api_token=api_token)

    def get_action(self, observations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get action prediction from remote server.
        
        Args:
            observations: Observation dictionary containing:
                - Video frames (RGB images)
                - Robot state (joint positions, etc.)
                - Task instruction (natural language)
                - Optional: prev_chunk and inference_delay for RTC mode
        
        Returns:
            Action dictionary with predicted actions:
                - Keys depend on policy configuration
                - For RTC mode, includes 'action_unified' for next query
        
        Example:
            >>> observations = {
            ...     'video.front_camera': np.array(...),  # (H, W, 3)
            ...     'state.joint_position': np.array(...),  # (7,)
            ...     'instruction': ["Pick up the red cube"]
            ... }
            >>> actions = client.get_action(observations)
            >>> print(actions['action.joint_position'])  # [[...], [...], ...]
        """
       
        print(f"📞 Calling 'get_action' on server {self.host}:{self.port}")
        return self.call_endpoint("get_action", observations)
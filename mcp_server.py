import sys
import json
from client import SACMaximumEntropyEvaluator

def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "genpark-sac-maximum-entropy-rl-evaluator-skill", "version": "1.0.0"}
            }
        }
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "tools": [
                    {
                        "name": "evaluate_sac_entropy",
                        "description": "Compute soft value function with Shannon entropy regularization and soft Bellman target",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "q_values": {"type": "array", "items": {"type": "number"}},
                                "policy_probs": {"type": "array", "items": {"type": "number"}},
                                "reward": {"type": "number", "default": 0.0},
                                "alpha": {"type": "number", "default": 0.2},
                                "gamma": {"type": "number", "default": 0.99}
                            },
                            "required": ["q_values", "policy_probs"]
                        }
                    }
                ]
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "evaluate_sac_entropy":
            qs = args.get("q_values", [])
            ps = args.get("policy_probs", [])
            r = args.get("reward", 0.0)
            a = args.get("alpha", 0.2)
            g = args.get("gamma", 0.99)
            sac = SACMaximumEntropyEvaluator(alpha=a, gamma=g)
            sv = sac.soft_value(qs, ps)
            target = sac.soft_bellman_target(r, sv["soft_value"])
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps({"soft_value": sv["soft_value"], "entropy": sv["entropy"], "soft_target": target})}]
                }
            }
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": "Method not found"}}

def main():
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            sys.stdout.write(json.dumps(res) + "\n")
            sys.stdout.flush()
        except Exception as e:
            err = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(e)}}
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()

if __name__ == "__main__":
    main()

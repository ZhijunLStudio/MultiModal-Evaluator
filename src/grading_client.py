import asyncio
import json
import time
import re
from typing import Dict, Any, List, Tuple, Optional
import aiohttp
from src.config import Config

class VerilogAGradingClient:
    def __init__(self, config: Config):
        self.config = config
        self.prompts = {}
        
    def extract_verilog_modules(self, verilog_code: str) -> List[Dict[str, Any]]:
        """Extract module definitions from Verilog-A code"""
        modules = []
        
        # Remove comments and clean up code
        cleaned_code = re.sub(r'//.*?\n', '\n', verilog_code)
        cleaned_code = re.sub(r'/\*.*?\*/', '', cleaned_code, flags=re.DOTALL)
        
        # Find all module definitions - more flexible pattern
        module_pattern = r'module\s+(\w+)(?:\s*#[^;]*?)?\s*\((.*?)\)\s*;(.*?)endmodule'
        matches = re.finditer(module_pattern, cleaned_code, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            module_name = match.group(1).strip()
            ports_str = match.group(2).strip()
            body = match.group(3).strip()
            
            # Parse ports
            ports = self.parse_ports(ports_str)
            
            modules.append({
                "name": module_name,
                "ports": ports,
                "body": body
            })
        
        return modules
    
    def parse_ports(self, ports_str: str) -> List[Dict[str, str]]:
        """Parse module ports with better flexibility"""
        if not ports_str.strip():
            return []
        
        ports = []
        # Split by comma but be careful with nested parentheses and brackets
        port_items = re.split(r',(?![^()\[\]]*[\)\]])', ports_str)
        
        for item in port_items:
            item = item.strip()
            if not item:
                continue
                
            # Match various port patterns:
            # input wire [3:0] port_name
            # input port_name  
            # output real port_name
            # inout [N-1:0] port_name
            port_match = re.match(r'(input|output|inout)?\s*(?:wire|reg|real|logic|electrical)?\s*(?:\[[^\]]+\])?\s*(\w+)', item.strip())
            if port_match:
                direction = port_match.group(1) if port_match.group(1) else "inout"
                name = port_match.group(2)
                ports.append({
                    "direction": direction,
                    "name": name
                })
        
        return ports
    
    def extract_instantiations(self, verilog_code: str) -> List[Dict[str, Any]]:
        """Extract module instantiations from Verilog-A code"""
        instantiations = []
        
        # Remove comments
        cleaned_code = re.sub(r'//.*?\n', '\n', verilog_code)
        cleaned_code = re.sub(r'/\*.*?\*/', '', cleaned_code, flags=re.DOTALL)
        
        # Find instantiations pattern: MODULE_NAME #(...) instance_name ( .port(signal), ... );
        # Handle parameterized modules too
        inst_pattern = r'(\w+)\s*(?:#[^(]*?)?\s+(\w+)\s*\(\s*(.*?)\s*\)\s*;'
        matches = re.finditer(inst_pattern, cleaned_code, re.DOTALL)
        
        for match in matches:
            module_type = match.group(1).strip()
            instance_name = match.group(2).strip()
            connections_str = match.group(3).strip()
            
            # Skip if it's a declaration, not an instantiation
            if module_type.lower() in ['electrical', 'integer', 'real', 'wire', 'reg', 'logic']:
                continue
                
            # Skip built-in Verilog-A constructs
            if module_type in ['V', 'I', 'idt', 'ddt', 'transition', 'cross']:
                continue
                
            # Parse connections
            connections = self.parse_connections(connections_str)
            
            instantiations.append({
                "module_type": module_type,
                "instance_name": instance_name,
                "connections": connections
            })
        
        return instantiations
    
    def parse_connections(self, connections_str: str) -> List[Dict[str, str]]:
        """Parse port connections in instantiation"""
        if not connections_str.strip():
            return []
        
        connections = []
        # Split by comma, handling nested parentheses
        conn_items = re.split(r',(?![^()]*\))', connections_str)
        
        for item in conn_items:
            item = item.strip()
            if not item:
                continue
                
            # Match .port(signal) pattern
            conn_match = re.match(r'\.(\w+)\s*\(\s*(\w+)\s*\)', item)
            if conn_match:
                port = conn_match.group(1)
                signal = conn_match.group(2)
                connections.append({
                    "port": port,
                    "signal": signal
                })
        
        return connections
    
    def normalize_module_name(self, name: str) -> str:
        """Normalize module names for comparison"""
        # Convert to lowercase and remove underscores/special chars for fuzzy matching
        return re.sub(r'[_\-\s]+', '', name.lower())
    
    def find_similar_modules(self, target_name: str, module_list: List[Dict]) -> List[str]:
        """Find modules with similar names"""
        target_norm = self.normalize_module_name(target_name)
        similar = []
        
        for module in module_list:
            module_norm = self.normalize_module_name(module["name"])
            
            # Exact match after normalization
            if target_norm == module_norm:
                similar.append(module["name"])
                continue
                
            # Check if one contains the other
            if target_norm in module_norm or module_norm in target_norm:
                similar.append(module["name"])
                continue
                
            # Check for semantic similarity (common mappings)
            mappings = {
                'adder': ['summer', 'add', 'sum'],
                'integrator': ['integration', 'int'],
                'comparator': ['comparison', 'comp'],
                'dac': ['digitaltoanalog', 'da'],
                'filter': ['decimation', 'decim'],
                'register': ['reg', 'ff', 'flipflop'],
                'logic': ['log'],
                'state': ['fsm', 'machine']
            }
            
            for key, aliases in mappings.items():
                if (key in target_norm and any(alias in module_norm for alias in aliases)) or \
                   (key in module_norm and any(alias in target_norm for alias in aliases)):
                    similar.append(module["name"])
                    break
        
        return similar
    
    def compare_components(self, generated_modules: List[Dict], reference_modules: List[Dict]) -> Dict[str, Any]:
        """Compare component modules between generated and reference code with fuzzy matching"""
        
        # Create lookup dictionaries
        gen_modules = {mod["name"]: mod for mod in generated_modules}
        ref_modules = {mod["name"]: mod for mod in reference_modules}
        
        correct_components = []
        missing_components = []
        extra_components = []
        
        # Check each reference module
        for ref_name, ref_mod in ref_modules.items():
            similar_modules = self.find_similar_modules(ref_name, generated_modules)
            
            if similar_modules:
                # Found at least one similar module
                best_match = similar_modules[0]  # Take the first/best match
                gen_mod = gen_modules[best_match]
                
                # Check if ports are reasonably similar (more lenient)
                if self.ports_reasonably_match(gen_mod["ports"], ref_mod["ports"]):
                    correct_components.append(f"{ref_name} -> {best_match}")
                else:
                    correct_components.append(f"{ref_name} -> {best_match} (partial)")
            else:
                missing_components.append(ref_name)
        
        # Find extra components in generated code
        matched_gen_modules = set()
        for ref_name in ref_modules:
            similar = self.find_similar_modules(ref_name, generated_modules)
            matched_gen_modules.update(similar)
        
        for gen_name in gen_modules:
            if gen_name not in matched_gen_modules:
                extra_components.append(gen_name)
        
        return {
            "correct_components": correct_components,
            "missing_components": missing_components,
            "extra_components": extra_components,
            "total_generated_components": len(generated_modules),
            "total_reference_components": len(reference_modules)
        }
    
    def ports_reasonably_match(self, gen_ports: List[Dict], ref_ports: List[Dict]) -> bool:
        """Check if port lists reasonably match (more lenient than exact match)"""
        if len(gen_ports) == 0 and len(ref_ports) == 0:
            return True
            
        # If both have ports, check for reasonable overlap
        if len(gen_ports) > 0 and len(ref_ports) > 0:
            gen_port_names = {self.normalize_module_name(p["name"]) for p in gen_ports}
            ref_port_names = {self.normalize_module_name(p["name"]) for p in ref_ports}
            
            # At least 50% overlap is considered reasonable
            overlap = len(gen_port_names.intersection(ref_port_names))
            max_ports = max(len(gen_port_names), len(ref_port_names))
            
            return overlap / max_ports >= 0.3  # 30% overlap threshold
        
        # If one has ports and other doesn't, it's a partial match
        return abs(len(gen_ports) - len(ref_ports)) <= 2
    
    def find_similar_instances(self, target_name: str, inst_list: List[Dict]) -> List[str]:
        """Find instances with similar names or module types"""
        similar = []
        target_norm = self.normalize_module_name(target_name)
        
        for inst in inst_list:
            inst_norm = self.normalize_module_name(inst["instance_name"])
            module_norm = self.normalize_module_name(inst["module_type"])
            
            # Check instance name similarity
            if target_norm == inst_norm or target_norm in inst_norm or inst_norm in target_norm:
                similar.append(inst["instance_name"])
                continue
                
            # Check module type similarity
            if target_norm == module_norm or target_norm in module_norm or module_norm in target_norm:
                similar.append(inst["instance_name"])
                continue
        
        return similar
    
    def compare_connections(self, generated_insts: List[Dict], reference_insts: List[Dict]) -> Dict[str, Any]:
        """Compare instantiations and connections with fuzzy matching"""
        
        gen_insts = {inst["instance_name"]: inst for inst in generated_insts}
        ref_insts = {inst["instance_name"]: inst for inst in reference_insts}
        
        correct_connections = []
        missing_connections = []
        extra_connections = []
        
        # Check each reference instantiation
        for ref_name, ref_inst in ref_insts.items():
            similar_instances = self.find_similar_instances(ref_name, generated_insts)
            
            if similar_instances:
                best_match = similar_instances[0]
                gen_inst = gen_insts[best_match]
                
                # Check if module types are similar
                if self.normalize_module_name(gen_inst["module_type"]) == self.normalize_module_name(ref_inst["module_type"]) or \
                   self.find_similar_modules(ref_inst["module_type"], [{"name": gen_inst["module_type"]}]):
                    
                    # Check connections similarity
                    if self.connections_reasonably_match(gen_inst["connections"], ref_inst["connections"]):
                        correct_connections.append(f"{ref_name} -> {best_match}")
                    else:
                        correct_connections.append(f"{ref_name} -> {best_match} (partial)")
                else:
                    missing_connections.append(f"{ref_name} (type mismatch)")
            else:
                missing_connections.append(ref_name)
        
        # Find extra instantiations
        matched_gen_insts = set()
        for ref_name in ref_insts:
            similar = self.find_similar_instances(ref_name, generated_insts)
            matched_gen_insts.update(similar)
        
        for gen_name in gen_insts:
            if gen_name not in matched_gen_insts:
                extra_connections.append(gen_name)
        
        return {
            "correct_connections": correct_connections,
            "missing_connections": missing_connections,
            "extra_connections": extra_connections,
            "total_generated_connections": len(generated_insts),
            "total_reference_connections": len(reference_insts)
        }
    
    def connections_reasonably_match(self, gen_connections: List[Dict], ref_connections: List[Dict]) -> bool:
        """Check if connections reasonably match"""
        if len(gen_connections) == 0 and len(ref_connections) == 0:
            return True
            
        if len(gen_connections) > 0 and len(ref_connections) > 0:
            # Normalize connection pairs for comparison
            gen_conn_set = {(self.normalize_module_name(c["port"]), self.normalize_module_name(c["signal"])) 
                           for c in gen_connections}
            ref_conn_set = {(self.normalize_module_name(c["port"]), self.normalize_module_name(c["signal"])) 
                           for c in ref_connections}
            
            # At least 30% overlap
            overlap = len(gen_conn_set.intersection(ref_conn_set))
            max_connections = max(len(gen_conn_set), len(ref_conn_set))
            
            return overlap / max_connections >= 0.3
        
        return abs(len(gen_connections) - len(ref_connections)) <= 2
    
    def calculate_scores(self, component_analysis: Dict, connection_analysis: Dict) -> Dict[str, float]:
        """Calculate component and connection scores with partial credit"""
        
        # Component score (0-50)
        total_ref_components = component_analysis["total_reference_components"]
        correct_components = len(component_analysis["correct_components"])
        
        if total_ref_components > 0:
            # Give partial credit for partial matches
            partial_credit = 0
            for component in component_analysis["correct_components"]:
                if "partial" in component:
                    partial_credit += 0.5
                else:
                    partial_credit += 1.0
            
            component_score = (partial_credit / total_ref_components) * 50
        else:
            component_score = 0
        
        # Connection score (0-50)
        total_ref_connections = connection_analysis["total_reference_connections"]
        correct_connections = len(connection_analysis["correct_connections"])
        
        if total_ref_connections > 0:
            # Give partial credit for partial matches
            partial_credit = 0
            for connection in connection_analysis["correct_connections"]:
                if "partial" in connection:
                    partial_credit += 0.5
                else:
                    partial_credit += 1.0
            
            connection_score = (partial_credit / total_ref_connections) * 50
        else:
            connection_score = 0
        
        # Total score
        total_score = component_score + connection_score
        
        return {
            "component_score": round(component_score, 1),
            "connection_score": round(connection_score, 1),
            "total_score": round(total_score, 1)
        }
    
    def analyze_verilog_a_code(self, generated_code: str, reference_code: str) -> Dict[str, Any]:
        """Analyze Verilog-A code and compare with reference"""
        
        try:
            # Extract components and connections
            generated_modules = self.extract_verilog_modules(generated_code)
            reference_modules = self.extract_verilog_modules(reference_code)
            
            generated_insts = self.extract_instantiations(generated_code)
            reference_insts = self.extract_instantiations(reference_code)
            
            # Compare components and connections
            component_analysis = self.compare_components(generated_modules, reference_modules)
            connection_analysis = self.compare_connections(generated_insts, reference_insts)
            
            # Calculate scores
            scoring = self.calculate_scores(component_analysis, connection_analysis)
            
            return {
                "component_analysis": component_analysis,
                "connection_analysis": connection_analysis,
                "scoring": scoring,
                "debug_info": {
                    "generated_modules": [m["name"] for m in generated_modules],
                    "reference_modules": [m["name"] for m in reference_modules],
                    "generated_instances": [f"{i['module_type']} {i['instance_name']}" for i in generated_insts],
                    "reference_instances": [f"{i['module_type']} {i['instance_name']}" for i in reference_insts]
                }
            }
        except Exception as e:
            import traceback
            return {
                "component_analysis": {"error": str(e)},
                "connection_analysis": {"error": str(e)},
                "scoring": {"component_score": 0, "connection_score": 0, "total_score": 0},
                "debug_info": {"error": f"{str(e)}\n{traceback.format_exc()}"}
            }
    
    async def grade(self, session: aiohttp.ClientSession, user_prompt: str, generated_answer: str, reference_answer: str) -> Dict[str, Any]:
        """Grade the generated Verilog-A code against reference"""
        
        start_time = time.time()
        
        # Always perform Verilog-A specific analysis first
        verilog_a_analysis = self.analyze_verilog_a_code(generated_answer, reference_answer)
        
        try:
            # Get grading prompt
            grading_prompt_key = "verilog_a_grading_prompt"
            if grading_prompt_key not in self.prompts:
                # Use a simplified prompt that just asks for feedback
                grading_prompt = """
                Please provide a brief technical assessment of the following Verilog-A code implementation:
                
                Reference Code:
                {reference_answer}
                
                Generated Code:
                {generated_answer}
                
                Focus on: 1) Structural correctness, 2) Functionality implementation, 3) Best practices.
                Provide feedback in 2-3 sentences.
                """
            else:
                grading_prompt = self.prompts[grading_prompt_key]
            
            # Format the prompt
            formatted_prompt = grading_prompt.format(
                reference_answer=reference_answer,
                generated_answer=generated_answer,
                user_prompt=user_prompt
            )
            
            # Prepare API request
            payload = {
                "model": self.config.grading_model,
                "messages": [
                    {
                        "role": "user", 
                        "content": formatted_prompt
                    }
                ],
                "max_tokens": self.config.grading_max_tokens or 1000,
                "temperature": self.config.grading_temperature or 0.3
            }
            
            headers = {
                "Authorization": f"Bearer {self.config.grading_api_key}",
                "Content-Type": "application/json"
            }
            
            # Construct API URL
            api_url = self.config.grading_api_base
            if not api_url.endswith('/'):
                api_url += '/'
            api_url += "chat/completions"
            
            # Make API request
            async with session.post(
                api_url,
                json=payload,
                headers=headers
            ) as response:
                
                response_data = await response.json()
                
                if response.status != 200:
                    error_msg = response_data.get("error", {}).get("message", "Unknown error")
                    llm_feedback = f"API Error: {error_msg}"
                    usage_info = {}
                else:
                    # Extract response content
                    llm_feedback = response_data["choices"][0]["message"]["content"]
                    
                    # Calculate token usage
                    usage_info = {
                        "grading_tokens": response_data.get("usage", {}).get("total_tokens", 0),
                        "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0)
                    }
        
        except Exception as e:
            llm_feedback = f"Failed to get LLM feedback: {str(e)}"
            usage_info = {}
        
        # Use our analysis scores (more reliable than LLM parsing)
        final_score = verilog_a_analysis["scoring"]["total_score"]
        
        return {
            "score": final_score,
            "content": llm_feedback,
            "usage": usage_info,
            "latency": time.time() - start_time,
            "verilog_a_analysis": verilog_a_analysis
        }
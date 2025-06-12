import os
import json
import re
import random
from collections import defaultdict
from openai import OpenAI # Import the OpenAI library
import traceback

# --- Configuration ---
# JSON_DIR should point to the directory containing your label data in JSON format
JSON_DIR = "./json_data_openai" # Example path for testing

# --- Class Definitions ---

class VerilogAParser:
    """Parses Verilog-A code to extract modules, ports, and instantiations."""

    def __init__(self):
        # 修改模块定义的正则表达式，支持两种格式：
        # 1. module name (port1, port2, ...);
        # 2. module name (input port1, output port2, ...);
        self.module_def_pattern = re.compile(
            r"module\s+([\w\d_]+)\s*\((.*?)\);(.*?)endmodule",
            re.DOTALL | re.IGNORECASE
        )
        
        # 修改端口匹配模式，支持两种格式：
        # 1. input/output real/logic/wire [vector] <name>
        # 2. input/output <name>
        # 3. <name> (不带input/output的端口)
        self.port_pattern = re.compile(
            r"(?:(?:\s*(input|output)\s+(?:real|logic|wire)?\s*(?:\[.*?\])?\s*([\w\d_]+))|(?:\s*(input|output)\s+([\w\d_]+))|(?:\s*([\w\d_]+)))",
            re.IGNORECASE
        )
        
        # 匹配模块体内部的input/output声明
        self.body_port_pattern = re.compile(
            r"(?:input|output)\s+(?:real|logic|wire)?\s*(?:\[.*?\])?\s*([\w\d_]+)(?:\s*,\s*([\w\d_]+))*",
            re.IGNORECASE
        )
        
        # 匹配electrical声明
        self.electrical_pattern = re.compile(
            r"electrical\s+([\w\d_]+(?:\s*,\s*[\w\d_]+)*)",
            re.IGNORECASE
        )
        
        # Regex for instantiations: Module_type Instance_name (.port(net), ...);
        self.instance_pattern = re.compile(
            r"^\s*([\w\d_]+)\s+([\w\d_]+)\s*\((.*?)\);",
            re.DOTALL | re.MULTILINE | re.IGNORECASE
        )
        
        # Regex for port connections: .InstancePort(NetName)
        self.port_connection_pattern = re.compile(
            r"\.(\w+)\s*\(([\w\d_\[\]:]+)\)",
            re.IGNORECASE
        )

    def parse(self, code):
        """Parses the Verilog-A code string."""
        modules_info = {}
        top_level_instantiations = []
        top_level_module_name = None

        raw_modules = []
        for match in self.module_def_pattern.finditer(code):
            raw_modules.append({
                "name": match.group(1),
                "ports_str": match.group(2),
                "body_str": match.group(3),
                "start_index": match.start()
            })

        if not raw_modules:
            return {"modules": {}, "top_level_module_name": None, "top_level_instantiations": [], "raw_code": code}

        # Determine top-level module
        if len(raw_modules) == 1:
            top_level_module_name = raw_modules[0]['name']
        else:
            potential_top_modules = {}
            module_names_defined = {m['name'] for m in raw_modules}
            last_module_in_file = raw_modules[-1]['name'] if raw_modules else None

            for m_data in raw_modules:
                score = 0
                for other_m_name in module_names_defined:
                    if other_m_name != m_data['name'] and \
                       re.search(r"\b" + re.escape(other_m_name) + r"\s+\w+", m_data['body_str']):
                        score += 1
                if "Top-Level" in m_data['name'] or \
                   "Top_Level" in m_data['name'] or \
                   (m_data['start_index'] > 0 and "Top-Level Interconnection" in code[max(0, m_data['start_index']-60):m_data['start_index']]):
                    score += 10
                if m_data['name'] == last_module_in_file and score == 0 and len(raw_modules) > 1:
                    score += 1
                if score > 0:
                    potential_top_modules[m_data['name']] = score
            
            if potential_top_modules:
                top_level_module_name = max(potential_top_modules, key=potential_top_modules.get)
            elif last_module_in_file:
                top_level_module_name = last_module_in_file
            else:
                top_level_module_name = raw_modules[0]['name']

        for m_data in raw_modules:
            module_name = m_data['name']
            parsed_ports = {
                "inputs": [],
                "outputs": [],
                "electrical": [],
                "ports": []  # 存储所有端口名，包括未指定方向的
            }
            
            # 首先处理模块定义行中的端口
            for port_match in self.port_pattern.finditer(m_data['ports_str']):
                # 处理三种格式的匹配结果
                if port_match.group(2):  # 第一种格式：input/output real/logic/wire [vector] <name>
                    direction, name = port_match.group(1), port_match.group(2)
                    parsed_ports[direction + "s"].append(name)
                    parsed_ports["ports"].append(name)
                elif port_match.group(4):  # 第二种格式：input/output <name>
                    direction, name = port_match.group(3), port_match.group(4)
                    parsed_ports[direction + "s"].append(name)
                    parsed_ports["ports"].append(name)
                elif port_match.group(5):  # 第三种格式：<name> (不带input/output)
                    name = port_match.group(5)
                    parsed_ports["ports"].append(name)

            # 处理模块体内部的input/output声明
            for body_port_match in self.body_port_pattern.finditer(m_data['body_str']):
                # 获取所有匹配的端口名
                ports = [p for p in body_port_match.groups() if p]
                # 根据声明类型（input/output）添加到相应的列表
                if "input" in body_port_match.group(0).lower():
                    parsed_ports["inputs"].extend(ports)
                elif "output" in body_port_match.group(0).lower():
                    parsed_ports["outputs"].extend(ports)
                parsed_ports["ports"].extend(ports)

            # 处理electrical声明
            electrical_match = self.electrical_pattern.search(m_data['body_str'])
            if electrical_match:
                electrical_ports = [p.strip() for p in electrical_match.group(1).split(',')]
                parsed_ports["electrical"].extend(electrical_ports)

            modules_info[module_name] = {"ports": parsed_ports, "type": "sub_module"}

            if module_name == top_level_module_name:
                modules_info[module_name]["type"] = "top_level"
                for inst_match in self.instance_pattern.finditer(m_data['body_str']):
                    inst_type, inst_name, inst_ports_str = inst_match.groups()
                    connections = {
                        conn_match.group(1): conn_match.group(2)
                        for conn_match in self.port_connection_pattern.finditer(inst_ports_str)
                    }
                    top_level_instantiations.append({
                        "instance_name": inst_name,
                        "module_type": inst_type,
                        "connections": connections
                    })
        
        return {
            "modules": modules_info,
            "top_level_module_name": top_level_module_name,
            "top_level_instantiations": top_level_instantiations,
            "raw_code": code
        }

    def get_module_signatures(self, parsed_code):
        """Extracts (name, num_inputs, num_outputs) for sub-modules."""
        signatures = {}
        for name, data in parsed_code.get("modules", {}).items():
            if data.get("type") == "sub_module":
                signatures[name] = (
                    len(data.get("ports", {}).get("inputs", [])),
                    len(data.get("ports", {}).get("outputs", []))
                )
        return signatures

    def get_port_lists(self, parsed_code, module_name):
        """Gets input and output port lists for a given module name."""
        module_data = parsed_code.get("modules", {}).get(module_name)
        if module_data:
            return {
                "inputs": module_data.get("ports", {}).get("inputs", []),
                "outputs": module_data.get("ports", {}).get("outputs", [])
            }
        return None

class LLMHelper:
    """Handles interaction with the OpenAI API for semantic mapping."""

    def __init__(self, api_key=None, model="o4-mini"):
        if api_key is None:
            api_key = "sk-EWCvOoyOvpl53kI3hYgFyq9vbyVWuefsWp9ODk2cYnIleDhA"
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key,base_url="https://jeniya.top/v1")

        # api_key = "EMPTY"
        # model = "Qwen2.5-VL-7B-Instruct"
        # self.client = OpenAI(api_key=api_key,base_url=f"http://192.168.99.119:8000/v1")
        self.model = model

    def _query_openai(self, prompt_text, system_message="You are an expert in Verilog-A and circuit design."):
        """Private method to send a query to OpenAI and get a JSON response."""
        print(f"\n--- Sending Query to OpenAI ({self.model}) ---")
        print(f"System: {system_message}")
        print(f"User Prompt (first 200 chars): {prompt_text[:200]}...")
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content":  prompt_text}
                ],
                response_format={"type": "json_object"}, # Requires newer models
                temperature=0.2 , # Lower temperature for more deterministic output
                stream=False
            )
            content = response.choices[0].message.content
            # print(f"--- OpenAI Response (Raw) ---\n{content}\n-----------------------------")
            if "```json" in content:
                content = re.sub(r'```json\n|\n```', '', content)
            else:
                content = content
            content = json.loads(content)
            # print(f"--- OpenAI Response (Parsed) ---\n{content}\n-----------------------------")
            return content
        except Exception as e:
            print(traceback.format_exc())
            print(f"Error querying OpenAI or parsing JSON response: {e}")
            return {} # Return empty dict on error

    def map_module_names(self, label_signatures, llm_signatures, label_top_module, llm_top_module):
        """Maps module names between label and LLM outputs using OpenAI."""
        prompt = f"""
            Analyze the following two sets of Verilog-A module signatures.
            Set 1 (Label): {json.dumps(label_signatures, indent=2)}
            Label's Top-Level Module: {label_top_module}

            Set 2 (LLM Output): {json.dumps(llm_signatures, indent=2)}
            LLM's Top-Level Module: {llm_top_module}

            Your task is to map module names from Set 1 (Label) to Set 2 (LLM Output).
            Prioritize matching I/O counts and semantically similar names (e.g., Adder to Summer, Integrator to Integration).
            The top-level modules ({label_top_module} and {llm_top_module}) should also be mapped if they appear in the signatures or as distinct top-level entities.
            Ensure the output is a single JSON object where keys are module names from Set 1 (Label) and values are the corresponding module names from Set 2 (LLM Output).
            Example: {{ "Adder_1": "Summer_1", "{label_top_module}": "{llm_top_module}"}}
            If a label module has no clear match, do not include it in the JSON.
            """
        return self._query_openai(prompt)

    def map_port_names(self, label_module_name, label_ports, llm_module_name, llm_ports):
        """Maps port names for a given pair of modules using OpenAI."""
        prompt = f"""
                Given two Verilog-A modules:
                1. Label Module: '{label_module_name}'
                Inputs: {json.dumps(label_ports.get('inputs',[]))}
                Outputs: {json.dumps(label_ports.get('outputs',[]))}

                2. LLM Module: '{llm_module_name}'
                Inputs: {json.dumps(llm_ports.get('inputs',[]))}
                Outputs: {json.dumps(llm_ports.get('outputs',[]))}

                Map the port names from the Label Module to the LLM Module. Consider semantic similarity (e.g., VIN to vin, Verr to vs) and port direction (input to input, output to output).
                Provide the mapping as a single JSON object where keys are port names from the Label Module and values are the corresponding port names from the LLM Module.
                Example: {{ "VIN": "vin", "Verr": "vs" }}
                If a label port has no clear match, do not include it in the JSON.
                """
        return self._query_openai(prompt)

class VerilogAComparator:
    """Compares two Verilog-A codes (label and LLM output) for modules and connections."""

    def __init__(self, parser: VerilogAParser, llm_helper: LLMHelper):
        self.parser = parser
        self.llm_helper = llm_helper


    def _build_connection_graph_2(self, parsed_code, module_mappings, port_mappings):
        """
        通过将端口映射到网络来构建一个规范化的连接图。
        这个修正后的版本能够准确地为电路的网络列表建模。
        """
        connections = set()
        top_module_name_original = parsed_code.get("top_level_module_name")
        instantiations = parsed_code.get("top_level_instantiations", [])

        if not top_module_name_original or not instantiations:
            return connections

        # --- 确定当前代码的上下文（是Label还是LLM），以便正确映射名称 ---
        is_llm_code_context = top_module_name_original in module_mappings.values()

        # 创建一个从LLM名称到Label名称的反向映射，方便查找
        llm_to_label_module_map = {v: k for k, v in module_mappings.items()}

        def get_label_module_name(original_name):
            if is_llm_code_context:
                return llm_to_label_module_map.get(original_name, original_name)
            return original_name

        def get_label_port_name(module_label_name, port_original_name):
            if is_llm_code_context:
                # 在LLM的上下文中，我们需要找到对应的label端口名
                llm_to_label_port_map = {v: k for k, v in port_mappings.get(module_label_name, {}).items()}
                return llm_to_label_port_map.get(port_original_name, port_original_name)
            # 在Label的上下文中，原始端口名就是label的端口名
            return port_original_name

        top_module_label_name = get_label_module_name(top_module_name_original)
        top_module_data = parsed_code["modules"].get(top_module_name_original)
        if not top_module_data:
            return connections

        # --- 构建网络列表的核心数据结构 ---
        # net_map[net_name] = {"driver": (模块类型, 端口名), "loads": [(模块类型, 端口名), ...]}
        net_map = defaultdict(lambda: {"driver": None, "loads": []})

        # 1. 将顶层输入视为驱动源
        for port_name_orig in top_module_data["ports"].get("inputs", []):
            port_label_name = get_label_port_name(top_module_label_name, port_name_orig)
            # 在顶层，网络名称就是端口本身的名称
            net_map[port_name_orig]["driver"] = (f"TOP_LEVEL_INPUT.{top_module_label_name}", port_label_name)

        # 2. 处理所有模块实例化，填充网络列表
        for inst in instantiations:
            inst_module_type_orig = inst["module_type"]
            inst_module_label_name = get_label_module_name(inst_module_type_orig)
            
            inst_module_def = parsed_code["modules"].get(inst_module_type_orig)
            if not inst_module_def:
                continue

            for inst_port_orig, net_name in inst["connections"].items():
                port_label_name = get_label_port_name(inst_module_label_name, inst_port_orig)
                
                # 检查端口是输出（驱动源）还是输入（负载）
                is_output_port = inst_port_orig in inst_module_def["ports"].get("outputs", [])
                # canonical_node = (inst_module_label_name, port_label_name)

                if is_output_port:
                    canonical_node = (inst_module_label_name, "output")
                    net_map[net_name]["driver"] = canonical_node
                else:
                    canonical_node = (inst_module_label_name, "input")
                    net_map[net_name]["loads"].append(canonical_node)
        
        # 3. 将顶层输出视为负载
        for port_name_orig in top_module_data["ports"].get("outputs", []):
            port_label_name = get_label_port_name(top_module_label_name, port_name_orig)
            # 顶层输出端口是连接到内部网络的一个负载
            canonical_node = (f"TOP_LEVEL_OUTPUT.{top_module_label_name}", port_label_name)
            net_map[port_name_orig]["loads"].append(canonical_node)

        # 4. 根据填充好的网络列表，构建最终的、规范化的连接图
        for net_name, drive_load_info in net_map.items():
            driver_info = drive_load_info["driver"]
            if not driver_info:
                # 跳过没有驱动源的网络 (例如，未连接的输入)
                continue

            for load_info in drive_load_info["loads"]:
                # 为驱动源和每个负载之间创建一条有向边
                connections.add((driver_info, load_info))
                
        return connections

    def _calculate_precision_recall_f1(self, tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1

    def run_comparison(self, label_code_str, llm_code_str):
        """Orchestrates the full comparison process."""
        results = {
            "module_metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            "connection_metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            "module_mappings": {},
            "port_mappings": {},
            "errors": []
        }
        print("label_code_str:",label_code_str)
        print("llm_code_str:",llm_code_str)
        parsed_label = self.parser.parse(label_code_str)
        parsed_llm = self.parser.parse(llm_code_str)

        if not parsed_label.get("modules"):
            results["errors"].append("Failed to parse label code.")
        if not parsed_llm.get("modules"):
            results["errors"].append("Failed to parse LLM code.")
        if results["errors"]:
            return results
        
        print("parsed_label:",parsed_label)
        print("parsed_llm:",parsed_llm)


        label_module_sigs = self.parser.get_module_signatures(parsed_label)
        llm_module_sigs = self.parser.get_module_signatures(parsed_llm)
        
        label_top = parsed_label.get("top_level_module_name", "UnknownLabelTop")
        llm_top = parsed_llm.get("top_level_module_name", "UnknownLLMTop")

        # Add top-level modules to signatures if not already sub-modules for mapping purposes
        if label_top not in label_module_sigs and label_top in parsed_label["modules"]:
            label_top_ports = self.parser.get_port_lists(parsed_label, label_top)
            if label_top_ports:
                 label_module_sigs[label_top] = (len(label_top_ports['inputs']), len(label_top_ports['outputs']))

        if llm_top not in llm_module_sigs and llm_top in parsed_llm["modules"]:
            llm_top_ports = self.parser.get_port_lists(parsed_llm, llm_top)
            if llm_top_ports:
                llm_module_sigs[llm_top] = (len(llm_top_ports['inputs']), len(llm_top_ports['outputs']))

        module_mappings = self.llm_helper.map_module_names(label_module_sigs, llm_module_sigs, label_top, llm_top)
        results["module_mappings"] = module_mappings

        port_mappings = {}
        for label_mod_name, llm_mod_name in module_mappings.items():
            if not label_mod_name or not llm_mod_name: continue
            label_ports = self.parser.get_port_lists(parsed_label, label_mod_name)
            llm_ports = self.parser.get_port_lists(parsed_llm, llm_mod_name)
            if label_ports and llm_ports:
                port_mappings[label_mod_name] = self.llm_helper.map_port_names(
                    label_mod_name, label_ports, llm_mod_name, llm_ports
                )
        results["port_mappings"] = port_mappings
        
        # Module Metrics
        label_sub_module_names = {m for m, d in parsed_label["modules"].items() if d.get("type") == "sub_module"}
        llm_actual_sub_modules = {m for m, d in parsed_llm["modules"].items() if d.get("type") == "sub_module"}
        
        tp_mod = 0
        mapped_llm_modules_in_tp = set()
        for label_mod in label_sub_module_names:
            llm_mapped_name = module_mappings.get(label_mod)
            if llm_mapped_name and llm_mapped_name in parsed_llm["modules"]:
                # Check I/O counts for structural similarity after LLM mapping
                label_io = (len(parsed_label["modules"][label_mod]["ports"]["inputs"]), len(parsed_label["modules"][label_mod]["ports"]["outputs"]))
                llm_io = (len(parsed_llm["modules"][llm_mapped_name]["ports"]["inputs"]), len(parsed_llm["modules"][llm_mapped_name]["ports"]["outputs"]))
                if label_io == llm_io:
                    tp_mod += 1
                    mapped_llm_modules_in_tp.add(llm_mapped_name)
        
        fp_mod = len(llm_actual_sub_modules - mapped_llm_modules_in_tp)
        fn_mod = len(label_sub_module_names) - tp_mod
        
        results["module_metrics"]["tp"] = tp_mod
        results["module_metrics"]["fp"] = fp_mod
        results["module_metrics"]["fn"] = fn_mod
        p, r, f1 = self._calculate_precision_recall_f1(tp_mod, fp_mod, fn_mod)
        results["module_metrics"]["precision"] = p
        results["module_metrics"]["recall"] = r
        results["module_metrics"]["f1"] = f1

        # Connection Metrics
        label_connections = self._build_connection_graph_2(parsed_label, module_mappings, port_mappings)
        llm_connections   = self._build_connection_graph_2(parsed_llm, module_mappings, port_mappings)
        
        tp_conn = len(label_connections.intersection(llm_connections))
        fp_conn = len(llm_connections - label_connections)
        fn_conn = len(label_connections - llm_connections)

        print("module mapping:")
        for k,v in module_mappings.items():
            print(k,v)


        print("port mapping:",)
        for  k,v in port_mappings.items():
            print(k,v)

        
        
        print("intersection nums:")
        for conn in label_connections.intersection(llm_connections):
            print(conn)
        print("fp_conn:")
        for conn in llm_connections - label_connections:
            print(conn)
        print("fn_conn:")
        for conn in label_connections - llm_connections:
            print(conn)

        results["connection_metrics"]["tp"] = tp_conn
        results["connection_metrics"]["fp"] = fp_conn
        results["connection_metrics"]["fn"] = fn_conn
        p, r, f1 = self._calculate_precision_recall_f1(tp_conn, fp_conn, fn_conn)
        results["connection_metrics"]["precision"] = p
        results["connection_metrics"]["recall"] = r
        results["connection_metrics"]["f1"] = f1

        print(results)

        return results
    
def get_llm_result(json_dir):
    """读取JSON文件中的每个问题的结果"""
    try:
        # 读取JSON文件
        with open(json_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        for item in data:
            # 提取对话内容
            conversations = item.get('conversations', [])
            if not conversations:
                continue
                
            # 获取最后一个GPT的回复（通常是Verilog-A代码）
            llm_code = None
            query= None
            for conv in reversed(conversations):
                if conv.get('from') == 'gpt':
                    llm_code = conv.get('value', '')

                if conv.get('from') == 'human':
                    query = conv.get('value', '')
            
            if not llm_code or not query:
                continue
                
            # 提取图片信息
            images = item.get('images', [])
            image_path = images[0] if images else None
            image_path= image_path.split("/")[-1]
            
            # 构建结果字典
            result = {
                'answer': llm_code,
                'images': image_path,
                'query': query
            }
            
            results[image_path]=result
        return results
        
    except Exception as e:
        print(f"读取JSON文件时发生错误: {str(e)}")
        return []
    

def get_benchmark(json_dir):
    """读取JSON文件中的每个问题的结果"""
    try:
        # 读取JSON文件
        with open(json_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        for item in data:
            answer = item.get("answer")
            image_path = item.get("img")
        
            # 构建结果字典
            result = {
                'answer': answer,
                'images': image_path,
            }
            
            results[image_path]=result
        return results
        
    except Exception as e:
        print(f"读取JSON文件时发生错误: {str(e)}")
        return [] 

# --- Main Execution (Test Code) ---
if __name__ == "__main__":
    
    try:
        # --- Initialization ---
        # Pass your OpenAI API Key here or ensure OPENAI_API_KEY env var is set
        # For model, you can use "gpt-4-turbo-preview" or other compatible models
        llm_helper = LLMHelper(model="gpt-4o") # Or "gpt-4o" / "gpt-4-turbo"
        parser = VerilogAParser()   
        comparator = VerilogAComparator(parser, llm_helper)

        # 获取所有LLM生成的结果
        llm_result = get_llm_result(".cache/converted_conversations.json")
        benchmark = get_benchmark(".cache/system_block_benchmark_v2_verilogA.json")
        test_list=["2747_block_circuit_train_15k_0321_001209.jpg"]
        # 处理每个结果
        for image_name, llm_info in llm_result.items():
            if image_name not in test_list:
                # print(image_name)
                continue 
            print(f"图片路径: {image_name}")
            if image_name not in benchmark:
                print(f"{image_name} not in benchmark.")
            comparator.run_comparison(benchmark[image_name]['answer'],llm_info['answer'])
            
    except Exception as e:
        print(traceback.format_exc())
        print(e)

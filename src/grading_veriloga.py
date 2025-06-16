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
        # 修改模块定义的正则表达式，支持多行格式和参数列表中的端口声明
        self.module_def_pattern = re.compile(
            r"module\s+([\w\d_]+)\s*\((.*?)\);(.*?)endmodule",
            re.DOTALL | re.IGNORECASE
        )
        
        # 更新端口声明模式，支持以下格式：
        # 1. input/output/inout <name>
        # 2. electrical <name>
        # 3. 支持多行声明
        # 4. 支持端口列表
        self.port_declaration_pattern = re.compile(
            r"^\s*(input|output|inout|electrical)\b\s+([^;]+);",
            re.DOTALL | re.MULTILINE | re.IGNORECASE
        )
        
        # 添加参数列表中的端口声明模式，支持类型声明和注释
        self.port_list_declaration_pattern = re.compile(
            r"(input|output|inout)\s+(?:real|integer|electrical|wire|reg)?\s*([\w\d_]+)",
            re.IGNORECASE
        )
        
        # 添加端口列表解析模式
        self.port_list_pattern = re.compile(
            r"([\w\d_]+)(?:\s*,\s*([\w\d_]+))*",
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

    def parse_ports(self, ports_str):
        """解析端口声明字符串，返回端口列表"""
        ports = []
        for line in ports_str.split(','):
            line = line.strip()
            if line:
                # 移除注释部分
                if '//' in line:
                    line = line.split('//')[0].strip()
                ports.extend([p.strip() for p in line.split(',')])
        return ports

    def parse(self, code):
        """Parses the Verilog-A code string."""
        modules_info = {}
        top_level_instantiations = []
        top_level_module_name = None

        raw_modules = []
        for match in self.module_def_pattern.finditer(code):
            module_name = match.group(1)
            ports_header = match.group(2)
            body_str = match.group(3)
            
            # 解析端口声明
            ports = {
                "inputs": [],
                "outputs": [],
                "inouts": [],
                "electrical": [],
                "all_ports": []
            }
            
            # 解析参数列表中的端口声明
            port_directions = {}  # 用于存储端口的方向信息
            port_types = {}       # 用于存储端口的类型信息
            
            # 处理参数列表中的端口声明
            # 首先移除注释
            ports_header_clean = re.sub(r'//.*$', '', ports_header, flags=re.MULTILINE)
            for port_decl in self.port_list_declaration_pattern.finditer(ports_header_clean):
                decl_type = port_decl.group(1).lower()
                port_name = port_decl.group(2).strip()
                
                port_directions[port_name] = decl_type
                if decl_type == "input":
                    ports["inputs"].append(port_name)
                elif decl_type == "output":
                    ports["outputs"].append(port_name)
                elif decl_type == "inout":
                    ports["inouts"].append(port_name)
                ports["all_ports"].append(port_name)
            
            # 处理模块体中的端口声明
            for port_decl in self.port_declaration_pattern.finditer(body_str):
                decl_type = port_decl.group(1).lower()
                port_names = self.parse_ports(port_decl.group(2))
                
                if decl_type in ["input", "output", "inout"]:
                    for port_name in port_names:
                        port_directions[port_name] = decl_type
                        if decl_type == "input":
                            ports["inputs"].append(port_name)
                        elif decl_type == "output":
                            ports["outputs"].append(port_name)
                        elif decl_type == "inout":
                            ports["inouts"].append(port_name)
                elif decl_type == "electrical":
                    for port_name in port_names:
                        port_types[port_name] = "electrical"
                        if port_name not in ports["electrical"]:
                            ports["electrical"].append(port_name)
            
            raw_modules.append({
                "name": module_name,
                "ports": ports,
                "body_str": body_str,
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

        # --- Module and Port Parsing ---
        for match in raw_modules:
            module_name = match['name']
            body_str = match['body_str']
            ports = match['ports']
            
            # 使用已经解析好的端口信息
            parsed_ports = ports
            
            modules_info[module_name] = {"ports": parsed_ports, "type": "sub_module"}

            if module_name == top_level_module_name:
                modules_info[module_name]["type"] = "top_level"
                for inst_match in self.instance_pattern.finditer(body_str):
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
        """Extracts (name, num_inputs, num_outputs, num_inouts) for sub-modules."""
        signatures = {}
        for name, data in parsed_code.get("modules", {}).items():
            if data.get("type") == "sub_module":
                signatures[name] = (
                    len(data.get("ports", {}).get("inputs", [])),
                    len(data.get("ports", {}).get("outputs", [])),
                    len(data.get("ports", {}).get("inouts", []))  # 添加inout端口数量
                )
        return signatures

    def get_port_lists(self, parsed_code, module_name):
        """Gets input, output, and inout port lists for a given module name."""
        module_data = parsed_code.get("modules", {}).get(module_name)
        if module_data:
            return {
                "inputs": module_data.get("ports", {}).get("inputs", []),
                "outputs": module_data.get("ports", {}).get("outputs", []),
                "inouts": module_data.get("ports", {}).get("inouts", [])  # 添加inout端口列表
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
        # print(f"\n--- Sending Query to OpenAI ({self.model}) ---")
        # print(f"System: {system_message}")
        # print(f"User Prompt (first 200 chars): {prompt_text[:200]}...")
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
            Each signature contains (num_inputs, num_outputs, num_inouts).
            Set 1 (Label): {json.dumps(label_signatures, indent=2)}
            Label's Top-Level Module: {label_top_module}

            Set 2 (LLM Output): {json.dumps(llm_signatures, indent=2)}
            LLM's Top-Level Module: {llm_top_module}

            Your task is to map module names from Set 1 (Label) to Set 2 (LLM Output).
            Prioritize matching I/O counts (including inout ports) and semantically similar names (e.g., Adder to Summer, Integrator to Integration).
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
                Inouts: {json.dumps(label_ports.get('inouts',[]))}

                2. LLM Module: '{llm_module_name}'
                Inputs: {json.dumps(llm_ports.get('inputs',[]))}
                Outputs: {json.dumps(llm_ports.get('outputs',[]))}
                Inouts: {json.dumps(llm_ports.get('inouts',[]))}

                Map the port names from the Label Module to the LLM Module. Consider:
                1. Semantic similarity (e.g., VIN to vin, Verr to vs)
                2. Port direction (input to input, output to output, inout to inout)
                3. Port type (electrical, real, etc.)
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
        这个修正后的版本能够准确地为电路的网络列表建模，并支持inout端口的双向连接。
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

        def is_driver_port(module_data, port_name):
            """判断一个端口是否可以作为驱动源"""
            return (port_name in module_data["ports"].get("outputs", []) or 
                   port_name in module_data["ports"].get("inouts", []))

        def is_load_port(module_data, port_name):
            """判断一个端口是否可以作为负载"""
            return (port_name in module_data["ports"].get("inputs", []) or 
                   port_name in module_data["ports"].get("inouts", []))

        top_module_label_name = get_label_module_name(top_module_name_original)
        top_module_data = parsed_code["modules"].get(top_module_name_original)
        if not top_module_data:
            return connections

        # --- 构建网络列表的核心数据结构 ---
        # net_map[net_name] = {"drivers": [(模块类型, 端口名), ...], "loads": [(模块类型, 端口名), ...]}
        net_map = defaultdict(lambda: {"drivers": [], "loads": []})

        # 1. 将顶层输入和inout视为驱动源
        for port_name_orig in top_module_data["ports"].get("inputs", []) + top_module_data["ports"].get("inouts", []):
            port_label_name = get_label_port_name(top_module_label_name, port_name_orig)
            net_map[port_name_orig]["drivers"].append((f"TOP_LEVEL_INPUT.{top_module_label_name}", port_label_name))

        # 2. 处理所有模块实例化，填充网络列表
        for inst in instantiations:
            inst_module_type_orig = inst["module_type"]
            inst_module_label_name = get_label_module_name(inst_module_type_orig)
            
            inst_module_def = parsed_code["modules"].get(inst_module_type_orig)
            if not inst_module_def:
                continue

            for inst_port_orig, net_name in inst["connections"].items():
                port_label_name = get_label_port_name(inst_module_label_name, inst_port_orig)
                
                # 检查端口类型并添加到相应的列表
                if is_driver_port(inst_module_def, inst_port_orig):
                    net_map[net_name]["drivers"].append((inst_module_label_name, port_label_name))
                if is_load_port(inst_module_def, inst_port_orig):
                    net_map[net_name]["loads"].append((inst_module_label_name, port_label_name))
        
        # 3. 将顶层输出和inout视为负载
        for port_name_orig in top_module_data["ports"].get("outputs", []) + top_module_data["ports"].get("inouts", []):
            port_label_name = get_label_port_name(top_module_label_name, port_name_orig)
            net_map[port_name_orig]["loads"].append((f"TOP_LEVEL_OUTPUT.{top_module_label_name}", port_label_name))

        # 4. 根据填充好的网络列表，构建最终的、规范化的连接图
        for net_name, drive_load_info in net_map.items():
            drivers = drive_load_info["drivers"]
            loads = drive_load_info["loads"]
            
            if not drivers or not loads:
                print(f"skip {net_name} because it has no driver or load")
                continue

            # 为每个驱动源和每个负载之间创建连接
            for driver in drivers:
                for load in loads:
                    if driver != load:  # 避免自连接
                        connections.add((driver, load))
                
        return connections

    def _calculate_precision_recall_f1(self, tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return precision, recall, f1
    

    def get_module_mappings_score(self,results):
        tp_mod = len(results['module_mappings'])
        fp_mod = len(results['module_mappings_fp'])
        fn_mod = len(results['module_mappings_fn'])
        
        results["module_mappings_metrics"]["tp"] = tp_mod
        results["module_mappings_metrics"]["fp"] = fp_mod
        results["module_mappings_metrics"]["fn"] = fn_mod
        p, r, f1 = self._calculate_precision_recall_f1(tp_mod, fp_mod, fn_mod)
        results["module_mappings_metrics"]["precision"]  = p
        results["module_mappings_metrics"]["recall"]     = r
        results["module_mappings_metrics"]["f1"]         = f1
        return results

    def get_module_score(self,results):
        tp_mod = len(results['match_modules'])
        fp_mod = len(results['FP_modules'])
        fn_mod = len(results['FN_modules'])
        
        results["module_metrics"]["tp"] = tp_mod
        results["module_metrics"]["fp"] = fp_mod
        results["module_metrics"]["fn"] = fn_mod
        p, r, f1 = self._calculate_precision_recall_f1(tp_mod, fp_mod, fn_mod)
        results["module_metrics"]["precision"]  = p
        results["module_metrics"]["recall"]     = r
        results["module_metrics"]["f1"]         = f1

        return results
    
    def get_connection_score(self,results):
        tp_conn = len(results['matched_connections'])
        fp_conn = len(results['FP_connections'])
        fn_conn = len(results['FN_connections'])

        # print("intersection nums:")
        # for conn in results['matched_connections']:
        #     print(conn)
        # print("fp_conn:")
        # for conn in results['FP_connections']:
        #     print(conn)
        # print("fn_conn:")
        # for conn in results['FN_connections']:
        #     print(conn)

        results["connection_metrics"]["tp"] = tp_conn
        results["connection_metrics"]["fp"] = fp_conn
        results["connection_metrics"]["fn"] = fn_conn
        p, r, f1 = self._calculate_precision_recall_f1(tp_conn, fp_conn, fn_conn)
        results["connection_metrics"]["precision"] = p
        results["connection_metrics"]["recall"] = r
        results["connection_metrics"]["f1"] = f1
        return results

    def get_module_match(self,parsed_label,parsed_llm,label_sub_module_names,llm_actual_sub_modules,module_mappings):

        tp_mod = 0
        mapped_llm_modules_in_tp = set()
        mapped_label_modules_in_tp = set()
        match_modules = []
        for label_mod in label_sub_module_names:
            llm_mapped_name = module_mappings.get(label_mod)
            if llm_mapped_name and llm_mapped_name in parsed_llm["modules"]:
                # Check I/O counts for structural similarity after LLM mapping
                label_io = (len(parsed_label["modules"][label_mod]["ports"]["inputs"]), len(parsed_label["modules"][label_mod]["ports"]["outputs"]),len(parsed_label["modules"][label_mod]["ports"]["inouts"]))
                llm_io = (len(parsed_llm["modules"][llm_mapped_name]["ports"]["inputs"]), len(parsed_llm["modules"][llm_mapped_name]["ports"]["outputs"]),len(parsed_llm["modules"][llm_mapped_name]["ports"]["inouts"]))
                if label_io == llm_io:
                    tp_mod += 1
                    mapped_llm_modules_in_tp.add(llm_mapped_name)
                    mapped_label_modules_in_tp.add(label_mod)
                    match_modules.append([label_mod,llm_mapped_name])
        return match_modules,mapped_llm_modules_in_tp,mapped_label_modules_in_tp

    def generate_html_report(self, results, image_name):
        """生成HTML格式的评估报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Verilog-A 评估报告</title>
            <style>
                body {{ 
                    font-family: Arial, sans-serif; 
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{ 
                    display: flex;
                    gap: 20px;
                    max-width: 1800px;
                    margin: 0 auto;
                }}
                .left-panel, .right-panel {{
                    flex: 1;
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    padding: 20px;
                    height: calc(100vh - 40px);
                    overflow-y: auto;
                }}
                .section {{ 
                    margin-bottom: 20px;
                    padding: 15px;
                    border: 1px solid #e0e0e0;
                    border-radius: 6px;
                    background: white;
                }}
                .section-title {{
                    font-size: 18px;
                    font-weight: bold;
                    margin-bottom: 15px;
                    color: #333;
                    border-bottom: 2px solid #eee;
                    padding-bottom: 8px;
                }}
                .code-block {{ 
                    background-color: #f8f8f8;
                    padding: 12px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    white-space: pre-wrap;
                    overflow-x: auto;
                    border: 1px solid #e0e0e0;
                }}
                .json-block {{
                    background-color: #f8f8f8;
                    padding: 12px;
                    border-radius: 4px;
                    font-family: 'Courier New', monospace;
                    font-size: 14px;
                    white-space: pre-wrap;
                    overflow-x: auto;
                    border: 1px solid #e0e0e0;
                }}
                .mapping-table {{ 
                    width: 100%;
                    border-collapse: collapse;
                    margin: 10px 0;
                    font-size: 14px;
                }}
                .mapping-table th, .mapping-table td {{ 
                    padding: 8px;
                    border: 1px solid #e0e0e0;
                    text-align: left;
                }}
                .mapping-table th {{ 
                    background-color: #f5f5f5;
                    font-weight: bold;
                }}
                .connection-list {{ 
                    list-style-type: none;
                    padding: 0;
                    margin: 0;
                }}
                .connection-item {{ 
                    padding: 8px;
                    margin: 4px 0;
                    background-color: #f8f8f8;
                    border-radius: 4px;
                    border: 1px solid #e0e0e0;
                    font-size: 14px;
                }}
                .success {{ color: #28a745; }}
                .warning {{ color: #ffc107; }}
                .error {{ color: #dc3545; }}
                .metric-box {{
                    display: inline-block;
                    padding: 8px 15px;
                    margin: 5px;
                    background-color: #f8f8f8;
                    border-radius: 4px;
                    border: 1px solid #e0e0e0;
                    font-size: 14px;
                }}
                .metric-value {{
                    font-weight: bold;
                    margin-right: 5px;
                }}
                .metric-label {{
                    color: #666;
                }}
                .module-mapping-section {{
                    margin-bottom: 20px;
                    padding: 15px;
                    background-color: #f8f8f8;
                    border-radius: 6px;
                    border: 1px solid #e0e0e0;
                }}
                .module-mapping-title {{
                    font-size: 16px;
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #333;
                }}
                .module-mapping-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    background-color: white;
                }}
                .module-mapping-table th, .module-mapping-table td {{
                    padding: 8px;
                    border: 1px solid #e0e0e0;
                    text-align: left;
                }}
                .module-mapping-table th {{
                    background-color: #f5f5f5;
                    font-weight: bold;
                }}
                .module-mapping-table td {{
                    font-family: 'Courier New', monospace;
                }}
                .connection-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 10px;
                    background-color: white;
                }}
                .connection-table th, .connection-table td {{
                    padding: 8px;
                    border: 1px solid #e0e0e0;
                    text-align: left;
                    font-family: 'Courier New', monospace;
                }}
                .connection-table th {{
                    background-color: #f5f5f5;
                    font-weight: bold;
                }}
                .port-info {{
                    margin-left: 20px;
                    font-size: 13px;
                    color: #666;
                }}
                .port-type {{
                    font-weight: bold;
                    color: #333;
                }}
                .unmatched-modules {{
                    display: flex;
                    gap: 20px;
                }}
                .unmatched-column {{
                    flex: 1;
                }}
                .module-pair {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 10px;
                }}
                .module-column {{
                    flex: 1;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="left-panel">
                    <div class="section">
                        <div class="section-title">原始代码</div>
                        <h3>标签代码</h3>
                        <div class="code-block">{label_code}</div>
                        <h3>LLM生成代码</h3>
                        <div class="code-block">{llm_code}</div>
                    </div>
                    <div class="section">
                        <div class="section-title">解析结果</div>
                        <h3>标签解析</h3>
                        <div class="json-block">{parsed_label}</div>
                        <h3>LLM解析</h3>
                        <div class="json-block">{parsed_llm}</div>
                    </div>
                </div>
                
                <div class="right-panel">
                    <div class="section">
                        <div class="section-title">评估指标</div>
                        <div class="metric-box">
                            <span class="metric-label">模块映射:</span>
                            <span class="metric-value">P: {module_mappings_precision:.2f}, R: {module_mappings_recall:.2f}, F1: {module_mappings_f1:.2f}</span>
                        </div>
                        <div class="metric-box">
                            <span class="metric-label">模块+端口数量匹配:</span>
                            <span class="metric-value">P: {module_precision:.2f}, R: {module_recall:.2f}, F1: {module_f1:.2f}</span>
                        </div>
                        <div class="metric-box">
                            <span class="metric-label">连接匹配:</span>
                            <span class="metric-value">P: {connection_precision:.2f}, R: {connection_recall:.2f}, F1: {connection_f1:.2f}</span>
                        </div>
                    </div>

                    <div class="section">
                        <div class="section-title">模块映射和匹配结果</div>
                        
                        <div class="module-mapping-section">
                            <div class="module-mapping-title">模块映射情况</div>
                            <table class="module-mapping-table">
                                <tr>
                                    <th>标签模块</th>
                                    <th>LLM模块</th>
                                </tr>
                                {module_mappings}
                            </table>
                        </div>

                        <div class="module-mapping-section">
                            <div class="module-mapping-title">模块匹配情况</div>
                            <table class="module-mapping-table">
                                <tr>
                                    <th>标签模块</th>
                                    <th>LLM模块</th>
                                </tr>
                                {module_matches}
                            </table>
                        </div>

                        <div class="module-mapping-section">
                            <div class="module-mapping-title">未匹配的模块</div>
                            <div class="unmatched-modules">
                                <div class="unmatched-column">
                                    <h3>标签模块</h3>
                                    {fn_modules}
                                </div>
                                <div class="unmatched-column">
                                    <h3>LLM模块</h3>
                                    {fp_modules}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="section">
                        <div class="section-title">连接信息</div>
                        <h3>标签连接</h3>
                        <table class="connection-table">
                            <tr>
                                <th>驱动源</th>
                                <th>负载</th>
                            </tr>
                            {label_connections}
                        </table>
                        <h3>LLM连接</h3>
                        <table class="connection-table">
                            <tr>
                                <th>驱动源</th>
                                <th>负载</th>
                            </tr>
                            {llm_connections}
                        </table>
                    </div>

                    <div class="section">
                        <div class="section-title">连接匹配结果</div>
                        <h3>正确匹配的连接</h3>
                        <ul class="connection-list">
                            {matched_connections}
                        </ul>

                        <h3>标签未匹配的连接</h3>
                        <ul class="connection-list">
                            {fn_connections}
                        </ul>

                        <h3>LLM未匹配的连接</h3>
                        <ul class="connection-list">
                            {fp_connections}
                        </ul>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

        # 准备模块映射表格内容
        module_mappings_html = ""
        for label_mod, llm_mod in results["module_mappings"].items():
            module_mappings_html += f"""
                <tr>
                    <td>{label_mod}</td>
                    <td>{llm_mod}</td>
                </tr>
            """

        # 准备模块匹配表格内容
        module_matches_html = ""
        for match in results["match_modules"]:
            label_mod, llm_mod = match
            module_matches_html += f"""
                <tr>
                    <td>{label_mod}</td>
                    <td>{llm_mod}</td>
                </tr>
            """

        # 准备未匹配和错误匹配的模块列表
        def format_module_with_ports(module_name, parsed_data):
            module_data = parsed_data["modules"].get(module_name, {})
            ports = module_data.get("ports", {})
            return f"""
                <div class='connection-item'>
                    {module_name}
                    <div class='port-info'>
                        <div><span class='port-type'>输入端口:</span> {', '.join(ports.get('inputs', []))}</div>
                        <div><span class='port-type'>输出端口:</span> {', '.join(ports.get('outputs', []))}</div>
                        <div><span class='port-type'>双向端口:</span> {', '.join(ports.get('inouts', []))}</div>
                    </div>
                </div>
            """

        # 创建映射关系字典
        label_to_llm = results["module_mappings"]
        llm_to_label = {v: k for k, v in label_to_llm.items()}

        # 准备未匹配模块的HTML
        def prepare_unmatched_modules_html():
            # 获取所有未匹配的模块
            fn_modules = set(results["FN_modules"])
            fp_modules = set(results["FP_modules"])
            
            # 创建模块对列表
            module_pairs = []
            
            # 1. 处理已映射但未匹配的模块对
            # 创建要移除的模块集合
            to_remove_label = set()
            to_remove_llm = set()
            
            for label_mod in fn_modules:
                if label_mod in label_to_llm:
                    llm_mod = label_to_llm[label_mod]
                    if llm_mod in fp_modules:
                        module_pairs.append((label_mod, llm_mod))
                        to_remove_label.add(label_mod)
                        to_remove_llm.add(llm_mod)
            
            # 移除已处理的模块
            fn_modules -= to_remove_label
            fp_modules -= to_remove_llm
            
            # 2. 处理剩余的未映射模块
            remaining_pairs = []
            for label_mod in sorted(fn_modules):
                remaining_pairs.append((label_mod, None))
            for llm_mod in sorted(fp_modules):
                remaining_pairs.append((None, llm_mod))
            
            # 生成HTML
            html = ""
            for label_mod, llm_mod in module_pairs + remaining_pairs:
                html += "<div class='module-pair'>"
                if label_mod:
                    html += f"<div class='module-column'>{format_module_with_ports(label_mod, results['parsed_label'])}</div>"
                else:
                    html += "<div class='module-column'></div>"
                if llm_mod:
                    html += f"<div class='module-column'>{format_module_with_ports(llm_mod, results['parsed_llm'])}</div>"
                else:
                    html += "<div class='module-column'></div>"
                html += "</div>"
            return html

        fn_modules_html = prepare_unmatched_modules_html()
        fp_modules_html = ""  # 不再需要单独的fp_modules_html，因为已经包含在fn_modules_html中

        # 准备连接列表内容
        def format_connection(conn):
            return f"<li class='connection-item'>{conn[0][0]}.{conn[0][1]} → {conn[1][0]}.{conn[1][1]}</li>"

        def format_connection_table(conn):
            return f"<tr><td>{conn[0][0]}.{conn[0][1]}</td><td>{conn[1][0]}.{conn[1][1]}</td></tr>"

        # 准备连接表格内容
        label_connections_html = "".join(format_connection_table(conn) for conn in results["label_connections"])
        llm_connections_html = "".join(format_connection_table(conn) for conn in results["llm_connections"])
        
        # 准备连接列表内容
        matched_connections_html = "".join(format_connection(conn) for conn in results["matched_connections"])
        fn_connections_html = "".join(format_connection(conn) for conn in results["FN_connections"])
        fp_connections_html = "".join(format_connection(conn) for conn in results["FP_connections"])

        # 格式化JSON输出
        parsed_label_str = json.dumps(results["parsed_label"], indent=2, ensure_ascii=False)
        parsed_llm_str = json.dumps(results["parsed_llm"], indent=2, ensure_ascii=False)

        # 填充模板
        html_content = html_template.format(
            image_name=image_name,
            module_mappings_f1=results["module_mappings_metrics"]["f1"],
            module_mappings_precision=results["module_mappings_metrics"]["precision"],
            module_mappings_recall=results["module_mappings_metrics"]["recall"],
            module_f1=results["module_metrics"]["f1"],
            module_precision=results["module_metrics"]["precision"],
            module_recall=results["module_metrics"]["recall"],
            connection_f1=results["connection_metrics"]["f1"],
            connection_precision=results["connection_metrics"]["precision"],
            connection_recall=results["connection_metrics"]["recall"],
            module_mappings=module_mappings_html,
            module_matches=module_matches_html,
            fn_modules=fn_modules_html,
            fp_modules=fp_modules_html,
            label_code=results["label_code_str"],
            llm_code=results["llm_code_str"],
            parsed_label=parsed_label_str,
            parsed_llm=parsed_llm_str,
            label_connections=label_connections_html,
            llm_connections=llm_connections_html,
            matched_connections=matched_connections_html,
            fn_connections=fn_connections_html,
            fp_connections=fp_connections_html
        )

        return html_content

    def run_comparison(self, label_code_str, llm_code_str):
        """Orchestrates the full comparison process."""
        results = {
            "module_metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            "connection_metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            "module_mappings_metrics": {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0},
            "label_code_str": label_code_str,
            "llm_code_str": llm_code_str,
            "parsed_label": {},
            "parsed_llm": {},
            "module_mappings": {},  # 模块映射结果
            "module_mappings_tp": {},  # 正确映射的模块
            "module_mappings_fp": [],  # 错误映射的模块
            "module_mappings_fn": [],  # 未映射的模块
            "match_modules": [],  # 匹配成功的模块
            "FP_modules": [],  # 错误匹配的模块
            "FN_modules": [],  # 未匹配的模块
            "matched_connections": [],  # 匹配成功的连接
            "FP_connections": [],  # 错误匹配的连接
            "FN_connections": [],  # 未匹配的连接
            "label_connections": [],  # 标签中的连接
            "llm_connections": [],  # LLM生成的连接
            "errors": []
        }

        parsed_label = self.parser.parse(label_code_str)
        parsed_llm = self.parser.parse(llm_code_str)

        if not parsed_label.get("modules"):
            results["errors"].append("Failed to parse label code.")
        if not parsed_llm.get("modules"):
            results["errors"].append("Failed to parse LLM code.")
        if results["errors"]:
            return results
        
        results['parsed_label'] = parsed_label
        results['parsed_llm'] = parsed_llm

        ## 模块映射
        label_module_sigs = self.parser.get_module_signatures(parsed_label)
        llm_module_sigs = self.parser.get_module_signatures(parsed_llm)
        
        label_top = parsed_label.get("top_level_module_name", "UnknownLabelTop")
        llm_top = parsed_llm.get("top_level_module_name", "UnknownLLMTop")

        # Add top-level modules to signatures if not already sub-modules for mapping purposes
        if label_top not in label_module_sigs and label_top in parsed_label["modules"]:
            label_top_ports = self.parser.get_port_lists(parsed_label, label_top)
            if label_top_ports:
                 label_module_sigs[label_top] = (
                     len(label_top_ports['inputs']),
                     len(label_top_ports['outputs']),
                     len(label_top_ports['inouts'])
                 )

        if llm_top not in llm_module_sigs and llm_top in parsed_llm["modules"]:
            llm_top_ports = self.parser.get_port_lists(parsed_llm, llm_top)
            if llm_top_ports:
                llm_module_sigs[llm_top] = (
                    len(llm_top_ports['inputs']),
                    len(llm_top_ports['outputs']),
                    len(llm_top_ports['inouts'])
                )

        # 获取模块映射结果
        module_mappings = self.llm_helper.map_module_names(label_module_sigs, llm_module_sigs, label_top, llm_top)

        print(module_mappings)
        results["module_mappings"] = module_mappings
        results["module_mappings_fp"] = list(set(llm_module_sigs.keys()) - set(module_mappings.values()))
        results["module_mappings_fn"] = list(set(module_mappings.keys()) - set(label_module_sigs.keys()))
        
        # 计算模块映射的精确率、召回率和F1值
        results = self.get_module_mappings_score(results)

        ## 模块匹配
        label_sub_module_names = {m for m, d in parsed_label["modules"].items() if d.get("type") == "sub_module"}
        llm_actual_sub_modules = {m for m, d in parsed_llm["modules"].items() if d.get("type") == "sub_module"}
        match_modules, mapped_llm_modules_in_tp, mapped_label_modules_in_tp = self.get_module_match(
            parsed_label, parsed_llm, label_sub_module_names, llm_actual_sub_modules, module_mappings
        )
        results['match_modules'] = match_modules
        results['FP_modules'] = list(llm_actual_sub_modules - mapped_llm_modules_in_tp)
        results['FN_modules'] = list(label_sub_module_names - mapped_label_modules_in_tp)
        results = self.get_module_score(results)

        ## 端口映射
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

        # Connection Metrics
        label_connections = self._build_connection_graph_2(parsed_label, module_mappings, port_mappings)
        llm_connections = self._build_connection_graph_2(parsed_llm, module_mappings, port_mappings)

        results['matched_connections'] = list(label_connections.intersection(llm_connections))
        results['FP_connections'] = list(llm_connections - label_connections)
        results['FN_connections'] = list(label_connections - llm_connections)
        results['label_connections'] = list(label_connections)
        results['llm_connections'] = list(llm_connections)
        
        # 计算连接的精确率、召回率和F1值
        results = self.get_connection_score(results)

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

        results_dir = ".cache/results"
        import shutil
        # if os.path.exists(results_dir):
        #     shutil.rmtree(results_dir)
        # os.makedirs(results_dir)

        # 获取所有LLM生成的结果
        llm_result = get_llm_result(".cache/converted_conversations.json")
        benchmark = get_benchmark(".cache/system_block_benchmark_v2_verilogA.json")
        # test_list=["14128_2078_block_circuit_train_15k_0321_001118.jpg"] #"2747_block_circuit_train_15k_0321_001209.jpg",
        # test_list = ["2695_block_circuit_train_15k_0321_001159.jpg"]
        test_list = ["2622_block_circuit_train_15k_0321_001159.jpg"]
        # test_list = ["12980_2061_block_circuit_train_15k_0321_001118.jpg"]
        # 处理每个结果
        count = 0
        for image_name, llm_info in llm_result.items():
            # if image_name not in test_list:
            #     # print(f"{image_name} not in test_list.")
            #     continue 
            json_path = f"{results_dir}/{image_name}.json"

            if os.path.exists(json_path):
                continue
            try:
                count+=1
                if count>20:
                    continue
                print(f"图片路径: {image_name}")
                if image_name not in benchmark:
                    print(f"{image_name} not in benchmark.")
                results = comparator.run_comparison(benchmark[image_name]['answer'],llm_info['answer'])

                # 保存JSON结果
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)

                # print(results)
                
                # 生成并保存HTML报告
                html_content = comparator.generate_html_report(results, image_name)
                with open(f"{results_dir}/{image_name}.html", "w", encoding="utf-8") as f:
                    f.write(html_content)
            except Exception as e:
                print(traceback.format_exc())
                print(e)
            
    except Exception as e:
        print(traceback.format_exc())
        print(e)

        print(e)

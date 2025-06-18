import re
import json

class VerilogParser:
    """
    A parser for Verilog and Verilog-A code to extract module information.
    """

    def __init__(self):
        """Initializes the parser with predefined regex patterns."""
        self.module_def_pattern = re.compile(
            r"module\s+([\w\d_]+)\s*(?:#\s*\(.*?\))?\s*\((.*?)\);(.*?)\bendmodule",
            re.DOTALL | re.IGNORECASE
        )
        self.port_declaration_pattern = re.compile(
            r"^\s*(input|output|inout|electrical)\b\s+([^;]+);",
            re.DOTALL | re.MULTILINE | re.IGNORECASE
        )
        
        # --- THIS IS THE CORRECTED LINE ---
        # Added \s* to allow whitespace between ')' and ';'
        self.instance_pattern = re.compile(
            r"^\s*([\w\d_]+)\s+([\w\d_]+)\s*\((.*?)\)\s*;",
            re.DOTALL | re.MULTILINE | re.IGNORECASE
        )
        
        self.port_connection_pattern = re.compile(
            r"\.([\w\d_]+)\s*\(\s*([\w\d_]+)\s*\)",
            re.DOTALL | re.IGNORECASE
        )

    def parse(self, code):
        """
        Parses the given Verilog/Verilog-A code string.
        """
        modules_info = {}
        raw_modules = list(self.module_def_pattern.finditer(code))

        if not raw_modules:
            return {"modules": {}, "top_level_module_name": None, "top_level_instantiations": []}

        module_names_defined = {m.group(1) for m in raw_modules}
        potential_top_modules = module_names_defined.copy()
        for m in raw_modules:
            body = m.group(3)
            for inst_match in self.instance_pattern.finditer(body):
                instantiated_type = inst_match.group(1)
                if instantiated_type in potential_top_modules:
                    potential_top_modules.remove(instantiated_type)
        
        if len(potential_top_modules) == 1:
            top_level_module_name = potential_top_modules.pop()
        elif raw_modules: # Fallback
            top_level_module_name = raw_modules[-1].group(1)
        else:
            top_level_module_name = None

        # --- Module and Port Parsing ---
        for match in raw_modules:
            module_name, ports_header, body_str = match.groups()
            
            parsed_ports = {"inputs": [], "outputs": [], "inouts": [], "electrical": [], "all_ports": []}
            port_directions = {}

            raw_header_ports = [p.strip() for p in ports_header.replace('\n', ' ').split(',') if p.strip()]
            for port_decl in raw_header_ports:
                parts = port_decl.split()
                if not parts: continue
                port_name = parts[-1]
                direction = parts[0] if len(parts) > 1 and parts[0] in ["input", "output", "inout"] else "unspecified"
                port_directions[port_name] = direction
                parsed_ports["all_ports"].append(port_name)

            for decl_match in self.port_declaration_pattern.finditer(body_str):
                direction, names_str = decl_match.groups()
                port_names = [p.strip() for p in names_str.split(',') if p.strip()]
                for port_name in port_names:
                    port_directions[port_name] = direction.lower()
                    if direction.lower() == 'electrical' and port_name not in parsed_ports['electrical']:
                        parsed_ports['electrical'].append(port_name)

            for name, direction in port_directions.items():
                if direction == "input": parsed_ports["inputs"].append(name)
                elif direction == "output": parsed_ports["outputs"].append(name)
                else: parsed_ports["inouts"].append(name)

            modules_info[module_name] = {"ports": parsed_ports, "type": "sub_module"}

        # --- Top-Level Instantiation Parsing ---
        top_level_instantiations = []
        if top_level_module_name and top_level_module_name in modules_info:
            modules_info[top_level_module_name]["type"] = "top_level"
            top_level_module_match = next((m for m in raw_modules if m.group(1) == top_level_module_name), None)
            if top_level_module_match:
                top_level_body = top_level_module_match.group(3)
                for inst_match in self.instance_pattern.finditer(top_level_body):
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
        }

# (The rest of the script to run and print the output remains the same)

# --- Main Execution ---
if __name__ == "__main__":
    verilog_code = """
    //node
    module Input_Device (
        output input_data
    );
    endmodule

    //node
    module Output_Device (
        input output_data
    );
    endmodule

    //node
    module I_O_Ports (
        input        input_data,
        input       output_data,
        inout        data_bus,
        inout        addr_bus,
        inout        ctrl_bus
    );
    endmodule

    //node
    module CPU (
        inout        data_bus,
        inout        addr_bus,
        inout        ctrl_bus
    );
    endmodule

    //node
    module MEMORY (
        inout        data_bus,
        inout        addr_bus,
        inout        ctrl_bus
    );
    endmodule


    //top-level
    module Top_Level (
        input  ext_input,
        output ext_output
    );
        // internal nets
        wire input_data_net;
        wire output_data_net;
        // ... (rest of the code)
        Input_Device   u_Input_Device   ( .input_data(input_data_net) );
        Output_Device  u_Output_Device  ( .output_data(output_data_net) );
    endmodule
    """

    veriloga_code = """
    `include "disciplines.vams"
    module InputDevice(io_pin);
        output io_pin;
        electrical io_pin;
    endmodule

    module OutputDevice(io_pin);
        input io_pin;
        electrical io_pin;
    endmodule
    
    module VonNeumann(primary_input, primary_output);
        input primary_input;
        output primary_output;
        electrical primary_input, primary_output;
        InputDevice input_dev(.io_pin(primary_input));
        OutputDevice output_dev(.io_pin(primary_output));
    endmodule
    """

    parser = VerilogParser()

    print("--- Parsing Verilog Code ---")
    parsed_verilog = parser.parse(verilog_code)
    print(json.dumps(parsed_verilog, indent=4))

    print("\n" + "---" * 20 + "\n")

    print("--- Parsing Verilog-A Code ---")
    parsed_veriloga = parser.parse(veriloga_code)
    print(json.dumps(parsed_veriloga, indent=4))
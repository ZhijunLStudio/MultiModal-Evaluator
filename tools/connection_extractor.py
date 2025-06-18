#!/usr/bin/env python
# connection_extractor.py - 连接关系提取器

import json
import re
import argparse
from typing import List, Tuple

def _extract_connections(text: str) -> List[Tuple[str, str, str]]:
    """Extract connection relationships from text using regex.
    Returns list of tuples (source, connection_type, target)
    """
    connections = []
    
    # 提取形式为 A -> B 的连接
    arrow_pattern = re.compile(r'(\w+)\s*->\s*(\w+)(?:\s*\[.*?\])?', re.DOTALL)
    matches = arrow_pattern.findall(text)
    for src, tgt in matches:
        connections.append((src.strip(), "->", tgt.strip()))
    
    # 提取形式为 A - B 或 A — B 的连接 (包括多种连字符，且处理可能没有空格的情况)
    dash_pattern = re.compile(r'(\w+)[^\w\n]*[—\-–−﹣－‐⁃‑‒\u2010-\u2015][^\w\n]*(\w+)(?:\s*\[.*?\])?', re.DOTALL)
    matches = dash_pattern.findall(text)
    for src, tgt in matches:
        connections.append((src.strip(), "-", tgt.strip()))
    
    # 提取形式为 A <-> B 的连接
    bidirectional_pattern = re.compile(r'(\w+)\s*<->\s*(\w+)(?:\s*\[.*?\])?', re.DOTALL)
    matches = bidirectional_pattern.findall(text)
    for src, tgt in matches:
        connections.append((src.strip(), "<->", tgt.strip()))
    
    return connections

def process_jsonl_file(file_path: str, check_field: str = "answer", verbose: bool = False):
    """
    处理JSONL文件，提取连接关系并打印结果
    
    Args:
        file_path: JSONL文件路径
        check_field: 要检查的字段名称（默认为"answer"）
        verbose: 是否打印详细信息
    """
    print(f"Processing file: {file_path}")
    print(f"Checking field: {check_field}")
    
    total_entries = 0
    empty_connections = 0
    successful_extractions = 0
    
    # 统计连接类型
    connection_types = {
        "->": 0,
        "-": 0,
        "<->": 0
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                try:
                    entry = json.loads(line)
                    total_entries += 1
                    
                    if check_field not in entry:
                        print(f"Warning: Entry at line {line_num} does not contain field '{check_field}'")
                        continue
                    
                    text = entry[check_field]
                    connections = _extract_connections(text)
                    
                    # 更新连接类型统计
                    for _, conn_type, _ in connections:
                        connection_types[conn_type] += 1
                    
                    if not connections:
                        empty_connections += 1
                        print(f"\n=== Empty connections at line {line_num} ===")
                        print(f"Image: {entry.get('img_folder', '')}/{entry.get('img', '')}")
                        print(f"Text sample: {text[:200]}{'...' if len(text) > 200 else ''}")
                    else:
                        successful_extractions += 1
                        if verbose:
                            print(f"\n--- Connections for image {entry.get('img', '')} (line {line_num}) ---")
                            for src, conn_type, tgt in connections:
                                print(f"  {src} {conn_type} {tgt}")
                
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON at line {line_num}")
                except Exception as e:
                    print(f"Error processing line {line_num}: {str(e)}")
        
        # 打印摘要统计
        print("\n=== Summary ===")
        print(f"Total entries processed: {total_entries}")
        print(f"Successful extractions: {successful_extractions} ({successful_extractions/total_entries*100:.1f}% of total)")
        print(f"Empty connection entries: {empty_connections} ({empty_connections/total_entries*100:.1f}% of total)")
        print(f"Connection types found:")
        print(f"  -> : {connection_types['->']}")
        print(f"  -  : {connection_types['-']}")
        print(f"  <->: {connection_types['<->']}")
        print(f"Total connections: {sum(connection_types.values())}")
        
        if empty_connections > 0:
            print(f"\nWarning: {empty_connections} entries had no connections extracted!")
            print("You may need to examine these entries and adjust the regular expressions.")
        
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
    except Exception as e:
        print(f"Error processing file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Extract connections from JSONL file")
    parser.add_argument("jsonl_file", help="Path to JSONL file")
    parser.add_argument("--field", default="answer", help="JSON field to check (default: answer)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print all connections")
    parser.add_argument("--analyze", "-a", action="store_true", help="Print detailed character analysis for empty connections")
    
    args = parser.parse_args()
    
    if args.analyze:
        # 添加详细字符分析功能
        def process_with_analysis(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        if not line.strip():
                            continue
                        
                        try:
                            entry = json.loads(line)
                            
                            if args.field not in entry:
                                continue
                            
                            text = entry[args.field]
                            connections = _extract_connections(text)
                            
                            if not connections:
                                print(f"\n=== Character Analysis for line {line_num} ===")
                                print(f"Image: {entry.get('img_folder', '')}/{entry.get('img', '')}")
                                
                                # 查找连字符
                                dashes = re.findall(r'[—\-–−﹣－‐⁃‑‒\u2010-\u2015]', text)
                                print(f"Found these dash characters: {[ord(d) for d in set(dashes)]}")
                                
                                # 打印每行的字符码点
                                lines = text.split('\n')
                                for i, line in enumerate(lines):
                                    if '—' in line or '-' in line:
                                        print(f"Line {i}: {line}")
                                        # 查找可能的连接
                                        print(f"Character codes: {[ord(c) for c in line]}")
                        
                        except json.JSONDecodeError:
                            print(f"Error: Could not parse JSON at line {line_num}")
                        except Exception as e:
                            print(f"Error analyzing line {line_num}: {str(e)}")
            
            except FileNotFoundError:
                print(f"Error: File {file_path} not found")
            except Exception as e:
                print(f"Error processing file: {str(e)}")
        
        process_with_analysis(args.jsonl_file)
    else:
        process_jsonl_file(args.jsonl_file, args.field, args.verbose)

if __name__ == "__main__":
    main()

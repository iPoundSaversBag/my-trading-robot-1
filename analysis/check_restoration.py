#!/usr/bin/env python3
"""
Check if the restored file is working properly
"""

def check_restored_file():
    file_path = '../plots_output/20250817_133240/performance_report.html'
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"File size: {len(content):,} chars")
    print(f"V4 dashboard present: {'v4-banner' in content.lower()}")
    print(f"QuantStats content present: {'quantstats' in content.lower()}")
    print(f"Container div present: {'<div class=\"container\"' in content}")
    print(f"HTML structure present: {'<html' in content.lower()}")
    print(f"Head section present: {'<head' in content.lower()}")
    
    # Check if V4 toolbar is present
    v4_toolbar = 'v4-toolbar' in content
    print(f"V4 toolbar present: {v4_toolbar}")
    
    # Check if QuantStats plots are present
    plots_present = 'plotly' in content.lower() or 'chart' in content.lower()
    print(f"Plots/charts present: {plots_present}")
    
    if len(content) > 25000 and v4_toolbar and plots_present:
        print("✅ File appears to be properly restored!")
        return True
    else:
        print("❌ File may still have issues")
        return False

if __name__ == '__main__':
    check_restored_file()

#!/usr/bin/env python3
"""
Analyze duplicate statistics contexts
"""
import re

def analyze_duplicates():
    file_path = '../plots_output/20250817_133240/performance_report.html'
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print("=== DUPLICATE STATISTICS ANALYSIS ===\n")
    
    # Find all Total Trades occurrences
    matches = list(re.finditer(r'Total Trades', content, re.IGNORECASE))
    
    for i, match in enumerate(matches):
        print(f"=== INSTANCE {i+1} (Position {match.start()}) ===")
        start = max(0, match.start() - 300)
        end = min(len(content), match.start() + 300)
        context = content[start:end]
        
        print(context)
        print("\n" + "="*60 + "\n")
        
        # Check if this is inside a specific section
        section_indicators = [
            'class="container"',
            'class="quick-stats"', 
            'Performance Summary',
            'Trading Statistics',
            'Risk Metrics'
        ]
        
        extended_start = max(0, match.start() - 1000)
        extended_context = content[extended_start:match.start()]
        
        print(f"Section context for instance {i+1}:")
        for indicator in section_indicators:
            if indicator in extended_context:
                last_pos = extended_context.rfind(indicator)
                if last_pos > -1:
                    section_snippet = extended_context[last_pos:last_pos+100]
                    print(f"  Found '{indicator}': ...{section_snippet}...")
        print()

if __name__ == '__main__':
    analyze_duplicates()

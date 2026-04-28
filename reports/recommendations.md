# Label QC Recommendations

- Dataset rows: **2776**
- EC classes: **6** | Substrate classes: **10** | Metal classes: **9**
- Missing rates: EC=0.36%, Substrate=41.43%, Metal=91.53%

## Action Items
- 金属依赖标签缺失率过高，建议优先补注“NONE/明确金属依赖”。
- 金属标签长尾较多（<8）：Ca2+, Ni2+, Co2+, Cu2+, Fe3+，建议保留主要金属种类。
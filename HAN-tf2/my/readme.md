### 文件说明

KG0：RPT+the same company+类别变量
KG1：RPT+previous year
KG2:RPT+previous year+类别变量+same manager
KG3:RPT+previous year+industry
RT_SC_adj(csv): RPT+ same company ;adj matrix for companies
RT_SC_B.npz:RPT+the same comp+the same manager ;adj sparse matrix for companies 
RT_SC_B_v2.npz:RPT+previous comp+the same manager ;adj sparse matrix for companies 

sameYearManager.csv:同一年是同一经理人的关系 m-m
sameManager.csv:同一经理人(不考虑经理人年份)的关系 m-m
sameYearIndustry.csv:同一年是同一行业 i-i
sameIndustry.csv:同一行业（不考虑年份） i-i

m:mamager,i:industry,rpt:related party transaction,y:year,py:previous year
HAN1：m-m，i-i，rpt，y-y,py
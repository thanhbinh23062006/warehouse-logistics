'use strict';
(() => {const O=DataCo;let data;
 function grouped(key,id){const rows=data.group_scores[key];if(!rows.length){O.$(id).parentElement.innerHTML='<p class="empty">Chưa có metadata '+O.esc(key)+'. Cần export bổ sung cột tương ứng.</p>';return;}
 const sorted=rows.map(g=>({name:g.name,rate:g.scores.filter(p=>p>=O.threshold).length/g.scores.length})).sort((a,b)=>key==='month'?a.name.localeCompare(b.name):b.rate-a.rate).slice(0,key==='month'?36:10);
 O.chart(id,key==='month'?'line':'bar',sorted.map(x=>x.name),[{label:'Tỷ lệ trễ dự báo (%)',data:sorted.map(x=>x.rate*100),backgroundColor:'#267d67',borderColor:'#267d67',borderRadius:5,tension:.3}]);}
 function render(){if(!data)return;
 const count=data.score_distribution.reduce((s,r)=>s+(r.probability>=O.threshold?r.count:0),0);
 O.$('total').textContent=O.fmt(data.total);O.$('lateRate').textContent=O.percent(count/data.total);O.$('highRisk').textContent=O.fmt(data.high_risk_orders);O.$('recall').textContent=O.decimal(O.config.metrics.recall);
 O.$('datasetNote').textContent=`Nguồn: delivery_predictions.csv · ${O.fmt(data.total)} dòng đã nạp. ${O.config.expected_rows==null?'Chưa có số dòng tham chiếu sau lọc.':data.total===O.config.expected_rows?'Khớp số dòng sau lọc trong notebook.':'Chưa khớp '+O.fmt(O.config.expected_rows)+' dòng sau lọc.'} Ngưỡng hiện tại ${O.decimal(O.threshold)}. Rủi ro cao tính theo order_id duy nhất, xác suất ≥ 70%.`;
 O.chart('histogram','bar',data.histogram.map(b=>`${Math.round(b.from*100)}–${Math.round(b.to*100)}%`),[{label:'Số dòng',data:data.histogram.map(b=>b.count),backgroundColor:data.histogram.map(b=>b.from>=O.threshold?'#d9a255':'#287e68'),borderRadius:4}]);
 grouped('category','categoryChart');grouped('month','monthChart');grouped('region','regionChart');
 O.$('topRows').innerHTML=data.top_orders.map(r=>`<tr><td><a href="order.html?id=${encodeURIComponent(r.order_id)}">${O.esc(r.order_id)}</a></td><td>${O.esc(r.order_item_id)}</td><td>${O.probability(r.xac_suat_tre)}</td><td>${O.label(r.xac_suat_tre)}</td><td><a href="order.html?id=${encodeURIComponent(r.order_id)}">Xem chi tiết ↗</a></td></tr>`).join('');
 }
 window.addEventListener('datacoready',async()=>{try{data=await O.api('/overview');render();}catch(e){O.error(e.message);}});window.addEventListener('thresholdchange',render);
 O.$('reloadOverview')?.addEventListener('click',()=>location.reload());
})();

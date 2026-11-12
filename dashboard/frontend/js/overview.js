'use strict';
(() => {const O=DataCo;let data,timer,sequence=0;
 function grouped(key,id){const group=data[key];if(!group.available){O.$(id).parentElement.innerHTML='<p class="empty">'+O.esc(group.reason)+'</p>';return;}
 const sorted=[...group.rows].sort((a,b)=>key==='by_month'?a.name.localeCompare(b.name):b.rate-a.rate).slice(0,key==='by_month'?36:10);
 O.chart(id,key==='by_month'?'line':'bar',sorted.map(x=>x.name),[{label:'Tỷ lệ trễ dự báo (%)',data:sorted.map(x=>x.rate*100),backgroundColor:'#267d67',borderColor:'#267d67',borderRadius:5,tension:.3}]);}
 function render(){if(!data)return;
 O.$('total').textContent=O.fmt(data.total);O.$('lateRate').textContent=O.percent(data.predicted_late_rate);O.$('highRisk').textContent=O.fmt(data.high_risk_orders);O.$('recall').textContent=O.decimal(O.config.metrics.recall);

 O.chart('histogram','bar',data.histogram.map(b=>`${Math.round(b.from*100)}–${Math.round(b.to*100)}%`),[{label:'Số dòng',data:data.histogram.map(b=>b.count),backgroundColor:data.histogram.map(b=>b.from>=data.threshold?'#d9a255':'#287e68'),borderRadius:4}]);
 grouped('by_category','categoryChart');grouped('by_month','monthChart');grouped('by_region','regionChart');
 O.$('topRows').innerHTML=data.top_orders.map(r=>`<tr><td><a href="order.html?id=${encodeURIComponent(r.order_id)}">${O.esc(r.order_id)}</a></td><td>${O.esc(r.order_item_id)}</td><td>${O.probability(r.xac_suat_tre)}</td><td>${O.label(r.xac_suat_tre)}</td><td><a href="order.html?id=${encodeURIComponent(r.order_id)}">Xem chi tiết ↗</a></td></tr>`).join('');
 }
 async function load(){const request=++sequence;O.$('reloadOverview').disabled=true;
  try{const result=await O.api('/overview?t='+encodeURIComponent(O.threshold));if(request!==sequence)return;data=result;O.error('');render();}
  catch(e){if(request===sequence)O.error(e.message);}
  finally{if(request===sequence)O.$('reloadOverview').disabled=false;}
 }
 window.addEventListener('datacoready',load);
 window.addEventListener('thresholdchange',()=>{++sequence;clearTimeout(timer);timer=setTimeout(load,180);});
 O.$('reloadOverview')?.addEventListener('click',load);
})();

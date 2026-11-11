'use strict';
(()=>{const O=DataCo;let timer,sequence=0;
 function renderMatrix(cm,source){O.$('confusion').innerHTML=['tn','fp','fn','tp'].map(k=>`<div><small>${k.toUpperCase()}</small><strong>${O.fmt(cm[k])}</strong><small>${({tn:'Đúng: không trễ',fp:'Báo nhầm trễ',fn:'Bỏ sót đơn trễ',tp:'Đúng: trễ'})[k]}</small></div>`).join('');O.$('cmSource').textContent=source;}
 async function refreshMatrix(){const request=++sequence;
  if(O.threshold==null)return;
  try{const data=await O.api('/evaluation?t='+encodeURIComponent(O.threshold));if(request!==sequence)return;if(data.available)renderMatrix(data.confusion_matrix,`Tập Test · xác suất có sẵn · ngưỡng ${O.decimal(data.threshold)}`);else{O.$('confusion').innerHTML='<p class="empty">'+O.esc(data.reason)+'</p>';O.$('cmSource').textContent='Chưa có đủ dữ liệu Test.';}}
  catch(e){if(request===sequence){O.$('confusion').innerHTML='<p class="empty">'+O.esc(e.message)+'</p>';O.$('cmSource').textContent='Chưa tải được ma trận ở ngưỡng hiện tại.';}}
 }
 window.addEventListener('thresholdchange',()=>{++sequence;clearTimeout(timer);O.$('cmSource').textContent='Đang cập nhật theo ngưỡng…';timer=setTimeout(refreshMatrix,180);});
 window.addEventListener('datacoready',()=>{const c=O.config;O.$('modelName').textContent=c.model;O.$('modelThreshold').textContent=O.decimal(c.threshold);O.$('featureCount').textContent=c.feature_count??'—';O.$('testMetrics').innerHTML=Object.entries(c.metrics).map(([name,value])=>`<div class="metric-small"><small>${O.esc(name.toUpperCase())} · tập test</small><strong>${O.decimal(value)}</strong></div>`).join('');
 if(c.validation_metrics)O.$('validationMetrics').innerHTML=Object.entries(c.validation_metrics).map(([name,value])=>`<div class="metric-small"><small>${O.esc(name.toUpperCase())} · Validation</small><strong>${O.decimal(value)}</strong></div>`).join('');
 const features=c.top_features;O.$('featureImportance').innerHTML=features.length?features.map(f=>`<div class="feature"><span>${O.esc(f.feature)}</span><div class="track"><i style="width:${Math.max(0,Math.min(100,f.importance*100))}%"></i></div><b>${Number(f.importance).toFixed(3)}</b></div>`).join(''):'<p class="empty">Chưa có feature importance từ artifact.</p>';
 if(c.monitoring_auc?.length){O.chart('aucChart','line',c.monitoring_auc.map(x=>x.period),[{label:'ROC-AUC trên nhóm Test theo tháng',data:c.monitoring_auc.map(x=>x.auc),borderColor:'#267d67',tension:.3}]);}else O.$('aucChart').parentElement.innerHTML='<p class="empty">Chưa có AUC đo theo tháng trên tập Test.</p>';
 if(c.confusion_matrix&&Math.abs(O.threshold-c.threshold)<1e-12)renderMatrix(c.confusion_matrix,'Tập Test · ngưỡng gốc '+O.decimal(c.threshold));else refreshMatrix();
 });
})();

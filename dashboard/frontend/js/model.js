'use strict';
(()=>{const O=DataCo;let scores=[];
 function renderMatrix(){const same=O.threshold!=null&&O.config.threshold!=null&&Math.abs(O.threshold-O.config.threshold)<1e-9;let cm=O.config.confusion_matrix;
 if((!same||!cm)&&scores.length){cm={tn:0,fp:0,fn:0,tp:0};for(const row of scores){cm[row.is_late?(row.probability>=O.threshold?'tp':'fn'):(row.probability>=O.threshold?'fp':'tn')]++;}}
 if((!same||!cm)&&!scores.length){O.$('confusion').innerHTML='<p class="empty">Chưa có nhãn thật is_late của tập test để tính ma trận tại ngưỡng này. Không nội suy từ ma trận gốc.</p>';return;}
 O.$('confusion').innerHTML=['tn','fp','fn','tp'].map(k=>`<div><small>${k.toUpperCase()}</small><strong>${O.fmt(cm[k])}</strong><small>${({tn:'Đúng: không trễ',fp:'Báo nhầm trễ',fn:'Bỏ sót đơn trễ',tp:'Đúng: trễ'})[k]}</small></div>`).join('');O.$('cmSource').textContent=same?'Tập test · cấu hình đã đối soát tại ngưỡng '+O.decimal(O.config.threshold):'Tính lại từ nhãn thật và xác suất tập test trong delivery_predictions.csv.';}
 window.addEventListener('thresholdchange',renderMatrix);
 window.addEventListener('datacoready',async()=>{const c=O.config;O.$('modelName').textContent=c.model;O.$('modelThreshold').textContent=O.decimal(c.threshold);O.$('featureCount').textContent=c.feature_count??'—';O.$('testMetrics').innerHTML=Object.entries(c.metrics).map(([name,value])=>`<div class="metric-small"><small>${name.toUpperCase()} · tập test</small><strong>${O.decimal(value)}</strong></div>`).join('');
 if(c.validation_metrics)O.$('validationMetrics').innerHTML=Object.entries(c.validation_metrics).map(([name,value])=>`<div class="metric-small"><small>${O.esc(name.toUpperCase())} · Validation</small><strong>${O.decimal(value)}</strong></div>`).join('');
 const features=c.top_features;O.$('featureImportance').innerHTML=features.length?features.map(f=>`<div class="feature"><span>${O.esc(f.feature)}</span><div class="track"><i style="width:${Math.max(0,Math.min(100,f.importance*100))}%"></i></div><b>${Number(f.importance).toFixed(3)}</b></div>`).join(''):'<p class="empty">Chưa có feature importance từ artifact. Không tạo trọng số giả.</p>';
 if(c.monitoring_auc?.length){O.chart('aucChart','line',c.monitoring_auc.map(x=>x.period),[{label:'ROC-AUC đo thực tế',data:c.monitoring_auc.map(x=>x.auc),borderColor:'#267d67',tension:.3}]);}else O.$('aucChart').parentElement.innerHTML='<p class="empty">Chưa có số liệu AUC đo theo thời gian.<br>Chạy cell export để lấy AUC theo tháng trên tập test thực tế.</p>';
 renderMatrix();try{const data=await O.api('/overview');scores=data.test_scores;renderMatrix();}catch{};
 });
})();

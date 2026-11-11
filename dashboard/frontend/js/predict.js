'use strict';
(()=>{const O=DataCo;let result;
 function render(){if(!result)return;const p=result.xac_suat_tre;O.$('predictionOutput').innerHTML=`<span class="tag">TỪ PIPELINE DATACO</span><div class="number">${O.percent(p)}</div>${O.label(p)}<div class="big-track"><i style="width:${p*100}%"></i></div><p class="muted">Ngưỡng hiện tại: ${O.decimal(O.threshold)} · Đổi nhãn tại chỗ.</p><h3>Yếu tố ảnh hưởng</h3>${result.top_ly_do.map(r=>`<div class="reason"><span>${O.esc(r.feature)}</span><b>${O.esc(r.anh_huong)} nguy cơ</b></div>`).join('')||'<p class="muted">'+O.esc(result.explanation_note||'Chưa có giải thích SHAP.')+'</p>'}<p class="source">Nguồn: model_tot_nhat.joblib · SHAP raw, không phải phần trăm xác suất.</p>`;}
 function fill(kind){const example=O.config?.demo_inputs?.[kind];if(!example){O.error('Chưa có mẫu '+kind+' được xác minh. Chạy cell xuất dashboard trong notebook DataCo.');return;}
 for(const [key,value] of Object.entries(example)){const field=O.$('predictForm').elements.namedItem(key);if(field)field.value=value;}O.error('');}
 O.$('sampleLate').onclick=()=>fill('late');O.$('sampleOnTime').onclick=()=>fill('on_time');
 window.addEventListener('datacoready',()=>{
  O.$('inputFields').innerHTML=O.config.form_fields.map(f=>{
   const attrs=`id="${O.esc(f.name)}" name="${O.esc(f.name)}" aria-describedby="error-${O.esc(f.name)}" ${f.required?'required':''}`;
   let input;
   if(f.type==='select'){input=`<select ${attrs}><option value="">${f.required?'Chọn giá trị':'Dùng mặc định Train'}</option>${(f.options||[]).map(v=>`<option value="${O.esc(v)}">${O.esc(v)}</option>`).join('')}</select>`;}
   else input=`<input ${attrs} type="${f.type}" ${f.min!=null?'min="'+f.min+'"':''} ${f.max!=null?'max="'+f.max+'"':''} ${f.type==='number'?'step="'+(f.integer?'1':'any')+'"':''} placeholder="${f.default!=null?'Mặc định: '+O.esc(f.default):''}">`;
   return `<div class="field"><label for="${O.esc(f.name)}">${O.esc(f.label)}${f.required?' *':''}</label>${input}<small>${f.required?'Bắt buộc nhập':'Để trống: dùng mặc định tính trên Train'}</small><span class="validation" id="error-${O.esc(f.name)}" aria-live="polite"></span></div>`;
  }).join('');O.$('predictSubmit').disabled=!O.config.ready;
 });
 O.$('predictForm').addEventListener('input',e=>{const parent=e.target.closest('.field');if(parent){parent.classList.remove('invalid');parent.querySelector('.validation').textContent='';}});
 O.$('predictForm').onsubmit=async e=>{e.preventDefault();const form=e.target;let valid=true;for(const field of form.querySelectorAll('input,select')){const parent=field.closest('.field'),bad=!field.checkValidity();parent.classList.toggle('invalid',bad);parent.querySelector('.validation').textContent=bad?'Hãy nhập đúng giá trị'+(field.step==='1'?' số nguyên':'')+'.':'';if(bad)valid=false;}if(!valid){form.querySelector('.invalid input,.invalid select')?.focus();return;}
 const payload=Object.fromEntries([...new FormData(form)].filter(([k,v])=>v!==''));O.$('predictSubmit').disabled=true;O.error('');
 try{result=await O.post('/predict',payload);render();}catch(err){O.error(err.message);}finally{O.$('predictSubmit').disabled=!O.config.ready;}};
 window.addEventListener('thresholdchange',render);
})();

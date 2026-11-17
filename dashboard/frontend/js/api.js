'use strict';
window.DataCo = (() => {
  const localPreview = ['localhost', '127.0.0.1'].includes(location.hostname) && location.port === '5500';
  const base = localPreview ? `${location.protocol}//${location.hostname}:8000/api` : '/api';
  const fmt = n => Number(n).toLocaleString('vi-VN');
  const percent = n => Number.isFinite(n) ? (n * 100).toLocaleString('vi-VN', {maximumFractionDigits:2}) + '%' : '—';
  const decimal = n => n == null ? '—' : Number(n).toFixed(4).replace('.', ',');
  const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const $ = id => document.getElementById(id);
  let config, charts = {};
  let t = null;
  async function api(path, options = {}) {
    const response = await fetch(base + path, {...options, signal: options.signal || AbortSignal.timeout(45000)});
    let data;
    try {data = await response.json();} catch {throw Error('Backend không trả JSON hợp lệ. Kiểm tra địa chỉ API.');}
    if (!response.ok) throw Error(typeof data.detail === 'string' ? data.detail : 'Yêu cầu chưa hợp lệ.');
    return data;
  }
  const post = (path, data) => api(path, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  function error(message, id='pageError') {const node=$(id); if(node){node.hidden=!message;node.textContent=message;}}
  function download(text, filename, type='text/csv;charset=utf-8') {
    const url=URL.createObjectURL(new Blob(['\ufeff'+text],{type}));const a=document.createElement('a');a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const label = p => t == null ? '<span class="label">Chưa có ngưỡng</span>' : `<span class="label ${p>=t?'late':''}">${p>=t?'TRỄ':'KHÔNG TRỄ'}</span>`;
  const probability = p => `<div class="probability"><span>${percent(p)}</span><div class="track"><i style="width:${p*100}%"></i></div></div>`;
  function chart(id, type, labels, datasets, options={}) {
    if(!window.Chart){const c=$(id);if(c)c.parentElement.textContent='Chưa tải được Chart.js. Kiểm tra kết nối mạng.';return;}
    if(charts[id])charts[id].destroy();
    const barLabels={id:'barValues',afterDatasetsDraw(chart){
      if(type!=='bar')return;
      const ctx=chart.ctx;ctx.save();ctx.fillStyle='#365267';ctx.font='600 10px sans-serif';ctx.textAlign='center';ctx.textBaseline='bottom';
      chart.data.datasets.forEach((dataset,index)=>{if(!chart.isDatasetVisible(index))return;
        chart.getDatasetMeta(index).data.forEach((bar,i)=>{const value=dataset.data[i];if(value==null||!Number.isFinite(Number(value)))return;
          const suffix=dataset.label?.includes('(%)')?'%':'';
          const text=Number(value).toLocaleString('vi-VN',{maximumFractionDigits:suffix?1:0})+suffix;
          const position=bar.tooltipPosition();ctx.fillText(text,position.x,Math.max(chart.chartArea.top+13,position.y-6));
        });
      });ctx.restore();
    }};
    charts[id]=new Chart($(id), {type,data:{labels,datasets},plugins:[barLabels],options:{responsive:true,maintainAspectRatio:false,layout:{padding:{top:type==='bar'?20:0}},plugins:{legend:{display:false}},scales:{x:{grid:{display:false},ticks:{font:{size:10},color:'#748497'}},y:{beginAtZero:true,grace:type==='bar'?'15%':0,grid:{color:'#edf1ef'},ticks:{font:{size:10},color:'#748497'}}},...options}});
  }
  function thresholdControl(){
    const input=$('threshold');if(!input)return;
    input.disabled=t==null; $('resetThreshold').disabled=t==null; input.value=t??.5; $('thresholdValue').textContent=decimal(t);
    input.addEventListener('input',()=>{t=Number(input.value);sessionStorage.setItem('dataco-threshold',String(t));$('thresholdValue').textContent=decimal(t);window.dispatchEvent(new CustomEvent('thresholdchange',{detail:t}));});
    $('resetThreshold').onclick=()=>{t=config.threshold;input.value=t;sessionStorage.setItem('dataco-threshold',String(t));$('thresholdValue').textContent=decimal(t);window.dispatchEvent(new CustomEvent('thresholdchange',{detail:t}));};
  }
  function initChat(){
    const toggle=$('chatToggle'), drawer=$('chatDrawer'),input=$('chatInput');let busy=false,history=[],predictionInput=null;
    window.addEventListener('predictionready',event=>{predictionInput=event.detail.input;});
    api('/chat-status').then(state=>{const label=$('chatState');if(label)label.textContent=state.configured?'Gemini đã cấu hình · theo dữ liệu đồ án':'Trợ lý tri thức · chưa cấu hình Gemini';}).catch(()=>{});
    function close(){drawer.hidden=true;toggle.setAttribute('aria-expanded','false');toggle.focus();}
    toggle.onclick=()=>{drawer.hidden=!drawer.hidden;toggle.setAttribute('aria-expanded',String(!drawer.hidden));if(!drawer.hidden)input.focus();};
    $('chatClose').onclick=close;document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!drawer.hidden)close();});
    function message(text,role,sources=''){const div=document.createElement('div');div.className='message '+role;div.textContent=text;if(sources){const small=document.createElement('small');small.textContent=sources;div.append(small);}$('chatMessages').append(div);div.scrollIntoView({block:'nearest'});return div;}
    async function send(text){if(busy||!text.trim())return;busy=true;input.value='';message(text,'user');const loading=message('Đang tra cứu tri thức đồ án…','ai');$('chatSend').disabled=true;
      try{const data=await post('/chat',{message:text,history:history.slice(-6),threshold:t,order_id:new URLSearchParams(location.search).get('id'),prediction_input:predictionInput});loading.remove();message(data.reply,'ai');history.push({role:'user',content:text},{role:'assistant',content:data.reply});}catch(e){loading.textContent=e.message;}finally{busy=false;$('chatSend').disabled=false;}}
    $('chatForm').onsubmit=e=>{e.preventDefault();send(input.value);};document.querySelectorAll('[data-question]').forEach(b=>b.onclick=()=>send(b.textContent));
  }
  async function init(){
    initChat();
    try{config=await api('/model-info');const saved=Number(sessionStorage.getItem('dataco-threshold'));t=config.threshold==null?null:(sessionStorage.getItem('dataco-threshold')!==null&&saved>=0&&saved<=1?saved:config.threshold);thresholdControl();$('connection').textContent=config.ready?'● Model đã nạp':'● Chờ artifacts';
      if(config.warnings.length)error(config.warnings.join(' '),'integrityWarning');
      if(config.artifact_error)error(config.artifact_error,'artifactWarning');
      window.dispatchEvent(new CustomEvent('datacoready',{detail:config}));
    }catch(e){error('Không kết nối được backend: '+e.message);$('connection').textContent='● Backend chưa sẵn sàng';}
  }
  document.addEventListener('DOMContentLoaded',init);
  return {api,post,fmt,percent,decimal,esc,$,error,download,label,probability,chart,get config(){return config;},get threshold(){return t;},base};
})();

'use strict';
(()=>{const O=DataCo;let data;
 function render(){if(!data)return;O.$('orderResult').hidden=false;O.$('orderId').textContent=data.order_id;O.$('orderProbability').textContent=O.percent(data.xac_suat_cap_don);O.$('orderLabel').innerHTML=O.label(data.xac_suat_cap_don);O.$('itemCount').textContent=O.fmt(data.items.length);O.$('orderRows').innerHTML=data.items.map(r=>`<tr><td>${O.esc(r.order_item_id)}</td><td>${O.esc(r.tap_du_lieu)}</td><td>${O.probability(r.xac_suat_tre)}</td><td>${O.label(r.xac_suat_tre)}</td></tr>`).join('');}
 async function lookup(id){if(!id.trim())return;O.error('');try{data=await O.api('/orders/'+encodeURIComponent(id));render();history.replaceState(null,'','?id='+encodeURIComponent(id));}catch(e){O.$('orderResult').hidden=true;O.error(e.message);}}
 O.$('orderLookup').onsubmit=e=>{e.preventDefault();lookup(O.$('lookupId').value);};window.addEventListener('thresholdchange',render);window.addEventListener('datacoready',()=>{const id=new URLSearchParams(location.search).get('id');if(id){O.$('lookupId').value=id;lookup(id);}});
})();

---
document_id: spec-cyber-banner.md
title: Banner Cyber Monday del año pasado
---

El banner promocional usado en la última edición de Cyber Monday en Marketplace Andes no está hardcodeado en el front permanente. Se sirve mediante feature flag `cyber_banner_legacy` (nombre ilustrativo en documentación interna), alineado con la regla de que cambios de Cyber Monday no quedan prendidos todo el año.

Cuando el flag está activo en la ventana de Cyber Monday, la home y listados muestran el arte y copy de la campaña anterior como referencia rápida para el equipo de producto. Fuera de flag, el banner no se renderiza aunque el asset exista en CDN.

Para el próximo Cyber Monday se espera nuevo creative; reutilizar el banner viejo es decisión explícita por flag, no comportamiento por defecto del checkout ni del catálogo.

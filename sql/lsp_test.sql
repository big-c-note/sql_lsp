select 
*
from system.query.history as h
where execution_status != "FINISHED"
limit 100

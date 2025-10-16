# %%
import etcd3
from store import PageData, open_for_write, commit, computed_field


cli = etcd3.client()

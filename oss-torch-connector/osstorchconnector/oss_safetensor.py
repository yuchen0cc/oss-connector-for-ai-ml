from ._oss_client import OssClient, DataObject
from ._oss_bucket_iterable import parse_oss_uri
from typing import Dict, Optional, Union, Any
import logging
import struct
import json
import ctypes
import torch
from safetensors.torch import _TYPES as _SAFETENSORS_TO_TORCH_DTYPE
from safetensors.torch import _SIZE as _TORCH_TO_SAFETENSORS_SIZE
_TORCH_TO_SAFETENSORS_DTYPE = {value: key for key, value in _SAFETENSORS_TO_TORCH_DTYPE.items()}

logger = logging.getLogger(__name__)


class oss_safe_open:
    def __init__(self, obj: DataObject, device: Union[str, int] = "cpu"):
        self._object = obj
        self._device = device
        header_len_bytes = self._object.read(8)
        if len(header_len_bytes) != 8:
            raise IOError("failed to read header length")

        header_len = struct.unpack('<Q', header_len_bytes)[0]

        header_json_bytes = self._object.read(header_len)
        if len(header_json_bytes) != header_len:
            raise IOError("failed to read header json")

        self._header = json.loads(header_json_bytes.decode('utf-8'))
        self._metadata = self._header.pop('__metadata__', {})
        self._data_start_offset = 8 + header_len

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        if _exc_type is not None:
            try:
                logger.info(
                    f"Exception occurred before closing safetensor: {_exc_type.__name__}: {_exc_value}"
                )
            except:
                pass
        else:
            self._object.close()

    def keys(self):
        """
        Returns the names of the tensors in the file.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        return list(self._header.keys())

    def offset_keys(self):
        """
        Returns the names of the tensors in the file, ordered by offset.

        Returns:
            (`List[str]`):
                The name of the tensors contained in that file
        """
        offset_sorted_keys = sorted(self._header.keys(), key=lambda k: self._header[k]['data_offsets'][0])
        return offset_sorted_keys

    def metadata(self):
        """
        Return the special non tensor information in the header

        Returns:
            (`Dict[str, str]`):
                The freeform metadata.
        """
        return self._metadata

    def get_tensor(self, name):
        """
        Returns a full tensor

        Args:
            name (`str`):
                The name of the tensor you want

        Returns:
            (`Tensor`):
                The tensor in the framework you opened the file for.
        """
        metadata = self._header.get(name)
        try:
            dtype_str = metadata['dtype']
            shape = metadata['shape']
            start, end = metadata['data_offsets']
        except KeyError as e:
            raise KeyError(f"failed to get tensor meta '{name}': {e}")

        dtype =_SAFETENSORS_TO_TORCH_DTYPE[dtype_str]
        tensor = torch.empty(shape, dtype=dtype, device=self._device)

        obj_offset = self._data_start_offset + start
        data_len = end - start
        buffer = (ctypes.c_char * data_len).from_address(tensor.data_ptr())

        self._object.seek(obj_offset)
        bytes_read = self._object.readinto(buffer)
        if bytes_read != data_len:
            raise IOError(f"failed to tensor")
        return tensor

    def get_slice(self, name):
        # Not implemented yet (less frequently used)
        raise NotImplementedError


class OssSafetensor:
    """A Safetensor manager for OSS.

        OssSafetensor provides an API and usage that are mostly consistent with Hugging Face's safetensors,
        allowing users to directly load tensors from or save tensors to OSS, bypassing the need for local file storage.
    """
    def __init__(
        self,
        endpoint: str,
        cred_path: str = "",
        config_path: str = "",
        cred_provider: Any = None,
        region: str = "",
    ):
        """
        Initialize an OssSafetensor for loading/saving safetensors.

        Args:
            endpoint(str): Endpoint of the OSS bucket where the objects are stored.
            cred_path(str): Credential info of the OSS bucket where the objects are stored.
            config_path(str): Configuration file path of the OSS connector.
            cred_provider: OSS credential provider.
            region(str): OSS region. Region will be inferred from 'endpoint' if not set, but this may fail when the endpoint lacks region information.
        """
        if not endpoint:
            raise ValueError("endpoint must be non-empty")
        else:
            self._endpoint = endpoint
        if not cred_path:
            self._cred_path = ""
        else:
            self._cred_path = cred_path
        if not config_path:
            self._config_path = ""
        else:
            self._config_path = config_path
        self._cred_provider = cred_provider
        self._region = region
        self._client = OssClient(self._endpoint, self._cred_path, self._config_path, cred_provider=self._cred_provider, region=self._region)

    def safe_open(self, oss_uri: str, device: Union[str, int] = "cpu") -> oss_safe_open:
        """Creates a safetensor object from a given oss_uri.
        Args:
            oss_uri (str): A valid oss_uri. (i.e. oss://<BUCKET>/<KEY>) which contains the tensors.
            device (`Union[str, int]`, *optional*, defaults to `cpu`):
                The device where the tensors need to be located after load.
                Available options are all regular torch device locations.
        """
        bucket, key = parse_oss_uri(oss_uri)
        obj = self._client.get_object(bucket, key, type=1)
        return oss_safe_open(obj, device=device)

    def load_file(
        self,
        oss_uri: str,
        device: Union[str, int] = "cpu"
    ) -> Dict[str, torch.Tensor]:
        """
        Loads a safetensors object on OSS into torch format.

        Args:
            oss_uri (str): A valid oss_uri. (i.e. oss://<BUCKET>/<KEY>) which contains the tensors.
            device (`Union[str, int]`, *optional*, defaults to `cpu`):
                The device where the tensors need to be located after load.
                Available options are all regular torch device locations.

        Returns:
            `Dict[str, torch.Tensor]`: dictionary that contains name as key, value as `torch.Tensor`
        """
        result = {}
        with self.safe_open(oss_uri, device=device) as f:
            for k in f.offset_keys():
                result[k] = f.get_tensor(k)
        return result

    def save_file(
        self,
        tensors: Dict[str, torch.Tensor],
        oss_uri: str,
        metadata: Optional[Dict[str, str]] = None,
    ):
        """
        Saves a dictionary of tensors into raw bytes in safetensors format on OSS.

        Args:
            tensors (`Dict[str, torch.Tensor]`):
                The incoming tensors. Tensors need to be contiguous and dense.
            oss_uri (`str`):
                A valid oss_uri. (i.e. oss://<BUCKET>/<KEY>), which is the object name we're saving into.
            metadata (`Dict[str, str]`, *optional*, defaults to `None`):
                Optional text only metadata you might want to save in your header.
                For instance it can be useful to specify more about the underlying
                tensors. This is purely informative and does not affect tensor loading.

        Returns:
            `None`

        """
        bucket, key = parse_oss_uri(oss_uri)
        obj = self._client.put_object(bucket, key)
        with obj as writer:
            self.do_save_safetensor(tensors, writer, metadata)

    def do_save_safetensor(self, tensors: Dict[str, torch.Tensor], obj: DataObject, metadata: Optional[Dict[str, str]] = None):
        header = {}
        current_offset = 0

        for name, tensor in tensors.items():
            if tensor.layout != torch.strided:
                raise ValueError(f"sparse tensor not support: `{name}`")

            if not tensor.is_contiguous():
                raise ValueError(f"non contiguous tensor not support: `{name}`")

            tensor_bytes = tensor.numel() * _TORCH_TO_SAFETENSORS_SIZE[tensor.dtype]
            header[name] = {
                "dtype": _TORCH_TO_SAFETENSORS_DTYPE.get(tensor.dtype),
                "shape": tensor.shape,
                "data_offsets": [current_offset, current_offset + tensor_bytes],
            }
            current_offset += tensor_bytes

        if metadata:
            if not all(isinstance(k, str) and isinstance(v, str) for k, v in metadata.items()):
                raise TypeError("Metadata keys and values must be strings.")
            header["__metadata__"] = metadata

        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")

        header_len_bytes = struct.pack("<Q", len(header_bytes))

        obj.write(header_len_bytes)
        obj.write(header_bytes)

        for name, tensor in tensors.items():
            data_offsets_list = header.get(name).get('data_offsets')
            tensor_bytes = data_offsets_list[1] - data_offsets_list[0]
            data = memoryview((ctypes.c_ubyte * tensor_bytes).from_address(tensor.data_ptr()))
            obj.write(data)
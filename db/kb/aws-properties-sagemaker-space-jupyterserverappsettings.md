# AWS::SageMaker::Space JupyterServerAppSettings<a name="aws-properties-sagemaker-space-jupyterserverappsettings"></a>

The JupyterServer app settings\.

## Syntax<a name="aws-properties-sagemaker-space-jupyterserverappsettings-syntax"></a>

To declare this entity in your AWS CloudFormation template, use the following syntax:

### JSON<a name="aws-properties-sagemaker-space-jupyterserverappsettings-syntax.json"></a>

```
{
  "[DefaultResourceSpec](#cfn-sagemaker-space-jupyterserverappsettings-defaultresourcespec)" : ResourceSpec
}
```

### YAML<a name="aws-properties-sagemaker-space-jupyterserverappsettings-syntax.yaml"></a>

```
  [DefaultResourceSpec](#cfn-sagemaker-space-jupyterserverappsettings-defaultresourcespec): 
    ResourceSpec
```

## Properties<a name="aws-properties-sagemaker-space-jupyterserverappsettings-properties"></a>

`DefaultResourceSpec`  <a name="cfn-sagemaker-space-jupyterserverappsettings-defaultresourcespec"></a>
The default instance type and the Amazon Resource Name \(ARN\) of the default SageMaker image used by the JupyterServer app\. If you use the `LifecycleConfigArns` parameter, then this parameter is also required\.  
*Required*: No  
*Type*: [ResourceSpec](aws-properties-sagemaker-space-resourcespec.md)  
*Update requires*: [No interruption](https://docs.aws.amazon.com/AWSCloudFormation/latest/UserGuide/using-cfn-updating-stacks-update-behaviors.html#update-no-interrupt)
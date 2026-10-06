// Regression test: serialization block copied verbatim from SampleApl.cpp.
// Hardware-free C++/CLI test; no spectrometer or motor SDK is linked.
#using <System.dll>
using namespace System;
using namespace System::Text;
using namespace System::Text::RegularExpressions;
int main() {
    String^ id = "serialization_regression";
    long result = 0;
    bool ok = true;
    array<unsigned short>^ data = gcnew array<unsigned short>(2048);
    for (int i = 0; i < data->Length; ++i) data[i] = static_cast<unsigned short>((i * 31) % 65536);
    data[2047] = 65535;
    StringBuilder^ text = gcnew StringBuilder();
text->Append("CAPTURE ")->Append(id)->Append(ok ? " OK 2048\n" : " FAIL " + result.ToString() + "\n");
        if (ok) {
            for each (unsigned short pixel in data) text->Append(pixel.ToString("X4"))->Append(" ");
            text->Append("\n");
        }
        text->Append("END ")->Append(id)->Append("\n");
        array<Byte>^ bytes = (gcnew System::Text::UTF8Encoding(false))->GetBytes(text->ToString());
    array<String^>^ tokens = Regex::Split(Encoding::UTF8->GetString(bytes)->Trim(), "\\s+");
    if (tokens->Length != 2054 || tokens[0] != "CAPTURE" || tokens[1] != id ||
        tokens[2] != "OK" || tokens[3] != "2048" || tokens[2052] != "END" || tokens[2053] != id) return 1;
    for (int i = 0; i < 2048; ++i) {
        if (!Regex::IsMatch(tokens[i+4], "^[0-9A-F]{4}$") ||
            Convert::ToUInt16(tokens[i+4], 16) != data[i]) return 2;
    }
    Console::WriteLine("PASS actual C++/CLI serialization: 2048 separate pixels; all values preserved; 2054 protocol tokens.");
    return 0;
}
